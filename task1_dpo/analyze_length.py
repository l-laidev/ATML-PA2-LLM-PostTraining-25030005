from __future__ import annotations

import argparse

import torch
import tqdm
from common.data import encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages, prompt_messages_from_preference, read_jsonl, repo_path
from common.generation import batch_generate, response_sequence_logprobs
from common.logging_utils import save_json
from common.metrics import mean_response_length, preference_accuracy, word_limit_compliance
from common.models import load_policy, load_tokenizer
from task1_dpo.train import make_collate, run_training

def stratified_evaluate(stratified, tokenizer, model, cfg):
    pref_accs = {}
    lengths = []
    mean_res_len = 0
    num_complied = 0
    num_total = 0
    for prompt, chosen, rejected, strata in tqdm.tqdm(stratified, desc="Stratified-length evaluation", total=len(stratified)):
        c = {k: v.to(model.device) for k,v in chosen.items()}
        r = {k: v.to(model.device) for k,v in rejected.items()}

        policy_chosen, _, _ = response_sequence_logprobs(model, c)
        policy_rejected, _, _ = response_sequence_logprobs(model, r)

        generation = batch_generate(
            model,
            tokenizer,
            prompt,
            cfg["max_sequence_length"],
            cfg["max_generation_tokens"],
            cfg["generation"]["temperature"],
            cfg["generation"]["top_p"],
            cfg["generation"]["do_sample"],
        )

        has_complied = 0
        for i in range(len(prompt)):
            cur_has_complied = word_limit_compliance(prompt[i][0]["content"], generation["responses"][i])
            cur_has_complied = 0 if cur_has_complied is None else int(cur_has_complied)
            has_complied += cur_has_complied
        has_complied /= len(prompt)

        lengths.extend(generation["response_lengths"])
        mean_res_len += mean_response_length(generation["response_mask"])
        num_complied += has_complied
        num_total += 1

        grouped = {}
        for i, stratum in enumerate(strata):
            grouped[stratum] = grouped.get(stratum, []) + [i]

        for stratum in grouped:
            sample_indices = grouped[stratum]
            pref_acc = preference_accuracy(policy_chosen[sample_indices], policy_rejected[sample_indices])
            pref_accs[stratum] = pref_accs.get(stratum, []) + [pref_acc]

    for stratum in pref_accs:
        pref_accs[stratum] = torch.as_tensor(pref_accs[stratum]).mean().item()

    std_res_len = torch.as_tensor(lengths, dtype=torch.float32).std().item()
    mean_res_len /= num_total
    num_complied /= num_total
    return {
        "pref_accs": pref_accs,
        "mean_res_len": mean_res_len,
        "std_res_len": std_res_len,
        "num_complied": num_complied,
    }

def compliance_evaluate(loader, tokenizer, model, cfg):
    lengths = []
    mean_res_len = 0
    num_complied = 0
    num_total = 0
    for prompt in tqdm.tqdm(loader, desc="Compliance evaluation", total=len(loader)):
        generation = batch_generate(
            model,
            tokenizer,
            prompt,
            cfg["max_sequence_length"],
            cfg["max_generation_tokens"],
            cfg["generation"]["temperature"],
            cfg["generation"]["top_p"],
            cfg["generation"]["do_sample"],
        )

        has_complied = 0
        for i in range(len(prompt)):
            cur_has_complied = word_limit_compliance(prompt[i][0]["content"], generation["responses"][i])
            cur_has_complied = 0 if cur_has_complied is None else int(cur_has_complied)
            has_complied += cur_has_complied
        has_complied /= len(prompt)

        lengths.extend(generation["response_lengths"])
        mean_res_len += mean_response_length(generation["response_mask"])
        num_complied += has_complied
        num_total += 1

    std_res_len = torch.as_tensor(lengths, dtype=torch.float32).std().item()
    mean_res_len /= num_total
    num_complied /= num_total
    return {
        "mean_res_len": mean_res_len,
        "std_res_len": std_res_len,
        "num_complied": num_complied,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))


    out = repo_path(cfg["length_output"])
    model_path = out / "length-balanced_model_final"
    if not model_path.exists():
        run_training(args.config, "length-balanced", cfg["paths"]["dpo_length_train"], out, cfg["beta"])
    else:
        print(f"Reusing model from: {model_path}")

    with torch.no_grad(), torch.inference_mode():
        tokenizer = load_tokenizer(cfg["base_model"])

        def custom_collate(rows, tokenizer):
            chosen, rejected = make_collate(tokenizer, int(cfg["max_sequence_length"]))(rows)
            return list(map(prompt_messages, rows)), chosen, rejected, [r["length_stratum"] for r in rows]

        def custom_collate_compliance(rows):
            return list(map(prompt_messages, rows))
        
        base_results = repo_path(cfg['results_dir'])
        model = load_policy(cfg, repo_path(cfg["standard_output"]) / "standard_model_final", trainable=False, fresh_lora=False)
        if not (base_results / "len-study-base.json").exists():
            loader = torch.utils.data.DataLoader(
                stratified,
                batch_size=8,
                shuffle=False,
                collate_fn=lambda rows: custom_collate(rows, tokenizer)
            )
            results = stratified_evaluate(loader, tokenizer, model, cfg)
            save_json(base_results / "len-study-base.json", results)

        if not (base_results / "compliance-base.json").exists():
            loader = torch.utils.data.DataLoader(
                read_jsonl(cfg["paths"]["word_limit_prompts"]),
                batch_size=8,
                shuffle=False,
                collate_fn=custom_collate_compliance
            )
            results = compliance_evaluate(loader, tokenizer, model, cfg)
            save_json(base_results / "compliance-base.json", results)

        model = load_policy(cfg, model_path, trainable=False, fresh_lora=False)
        if not (base_results / "len-study-trained.json").exists():
            # recreate to ensure same conditions
            loader = torch.utils.data.DataLoader(
                stratified,
                batch_size=8,
                shuffle=False,
                collate_fn=lambda rows: custom_collate(rows, tokenizer)
            )
            results = stratified_evaluate(loader, tokenizer, model, cfg)
            save_json(base_results / "len-study-trained.json", results)

        if not (base_results / "compliance-trained.json").exists():
            loader = torch.utils.data.DataLoader(
                read_jsonl(cfg["paths"]["word_limit_prompts"]),
                batch_size=8,
                shuffle=False,
                collate_fn=custom_collate_compliance
            )
            results = compliance_evaluate(loader, tokenizer, model, cfg)
            save_json(base_results / "compliance-trained.json", results)

if __name__ == "__main__":
    main()
