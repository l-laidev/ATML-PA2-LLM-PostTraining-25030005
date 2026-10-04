from __future__ import annotations

import argparse

import torch
import tqdm
from common.data import encode_prompt_response, load_yaml, pad_batch, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
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
            [prompt],
            cfg["max_sequence_length"],
            cfg["max_generation_tokens"],
            cfg["generation"]["temperature"],
            cfg["generation"]["top_p"],
            cfg["generation"]["do_sample"],
        )

        generation_text = tokenizer.batch_decode(generation["response_ids"], skip_special_tokens=True)
        has_complied = 0
        for i in range(len(prompt)):
            cur_has_complied = word_limit_compliance(prompt, generation_text)
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
    run_training(args.config, "length-balanced", cfg["paths"]["dpo_length_train"], out / "length-balanced", cfg["beta"])

# {"source_split": "test_prefs", "source_index": 1858, 
# "prompt_id": "455c67a88d63f1db574d7117e7cc1dcd8aec226536d9f041cee97b2462ff7473", 
# "prompt": "Given the text: Our gentle, step-by-step method of instruction is designed to help children (and sometimes adults) bloom creatively while learning the fundamentals of ART, WRITING or SPANISH in a calm, friendly environment; free of judgment and competition.\nAfterschool Art and More is a LOCALLY OWNED & OPERATED grassroots organization, dedicated to providing our community with high quality, personal, and professional enrichment programming in ART, WRITING and SPANISH.\nYour support is vital and appreciated.\nHow does Afterschool Art and More contribute to the community as a grassroots organization?", 
# "chosen": [{"content": "Given the text: Our gentle, step-by-step method of instruction is designed to help children (and sometimes adults) bloom creatively while learning the fundamentals of ART, WRITING or SPANISH in a calm, friendly environment; free of judgment and competition.\nAfterschool Art and More is a LOCALLY OWNED & OPERATED grassroots organization, dedicated to providing our community with high quality, personal, and professional enrichment programming in ART, WRITING and SPANISH.\nYour support is vital and appreciated.\nHow does Afterschool Art and More contribute to the community as a grassroots organization?", 
#           "role": "user"}, 
#           {"content": "Afterschool Art and More contributes to the community as a grassroots organization by focusing on providing high-quality, personal, and professional enrichment programs in art, writing, and Spanish for children and adults. They create a calm, friendly environment free of judgment and competition, which encourages creative growth and learning. As a locally owned and operated business, they are dedicated to serving their community's needs, and support from the community is vital for their continued work and success.", 
#           "role": "assistant"}], 
# "rejected": [{"content": "Given the text: Our gentle, step-by-step method of instruction is designed to help children (and sometimes adults) bloom creatively while learning the fundamentals of ART, WRITING or SPANISH in a calm, friendly environment; free of judgment and competition.\nAfterschool Art and More is a LOCALLY OWNED & OPERATED grassroots organization, dedicated to providing our community with high quality, personal, and professional enrichment programming in ART, WRITING and SPANISH.\nYour support is vital and appreciated.\nHow does Afterschool Art and More contribute to the community as a grassroots organization?", 
#           "role": "user"}, 
#           {"content": "Afterschool Art and More contributes to the community as a grassroots organization by providing high quality, personal, and professional enrichment programming in ART, WRITING, and SPANISH. By offering a calm, friendly environment that is free of judgment and competition, the organization helps children and sometimes adults bloom creatively while learning the fundamentals of these subjects. Additionally, as a LOCALLY OWNED & OPERATED organization, Afterschool Art and More is committed to serving the needs of its community, and its support is vital and appreciated. Overall, the organization's grassroots approach allows it to have a positive impact on the community it serves.", 
#           "role": "assistant"}], 
# "score_chosen": 8.5, 
# "score_rejected": 8.5, 
# "length_stratum": "rejected_longer", 
# "chosen_tokens": 92, 
# "rejected_tokens": 129, 
# "length_difference": -37}

    with torch.no_grad(), torch.inference_mode():
        tokenizer = load_tokenizer(cfg["base_model"])

        def custom_collate(rows, tokenizer):
            chosen, rejected = make_collate(tokenizer, int(cfg["max_sequence_length"]))(rows)
            return [r["prompt"] for r in rows], chosen, rejected, [r["length_stratum"] for r in rows]
        loader = torch.utils.data.DataLoader(
            stratified,
            batch_size=8,
            shuffle=False,
            collate_fn=lambda rows: custom_collate(rows, tokenizer)
        )
        
        model = load_policy(cfg, cfg["standard_output"], trainable=False, fresh_lora=False)
        results = stratified_evaluate(loader, tokenizer, model, cfg)
        save_json(f"{cfg['results_dir']}/len-study-base.json", results)
        
        model = load_policy(cfg, out / "length-balanced", trainable=False, fresh_lora=False)
        results = stratified_evaluate(loader, tokenizer, model, cfg)
        save_json(f"{cfg['results_dir']}/len-study-trained.json", results)

if __name__ == "__main__":
    main()
