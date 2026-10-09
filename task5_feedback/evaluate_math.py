from __future__ import annotations

import argparse
from pathlib import Path

import torch
import tqdm

from common.data import load_yaml, read_jsonl
from common.generation import batch_generate
from common.logging_utils import save_json
from common.metrics import mean_response_length
from common.models import load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    args = ap.parse_args()
    cfg, rows, _ = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    print("Exact verifier available as task5_feedback.rlvr.exact_reward")
    print("Pairwise judge available as task5_feedback.rlaif.PairwiseAIJudge")
    
    responses = {}
    for policy_name in policy_specs(cfg).keys():
        print(f"Extracting responses from: {policy_name}")
        
        policy = load_frozen_policy(cfg, policy_name)
        tokenizer = load_tokenizer(cfg["base_model"])
        responses[policy_name] = {}
        
        # source_index, question, gold_solution, gold_final, message, prompt
        for row in tqdm.tqdm(rows, desc="Generating"):
            responses[policy_name][row["source_index"]] = {
                "generation": batch_generate(
                    policy,
                    tokenizer,
                    [row["message"]],
                    max_prompt_length=None,
                    max_new_tokens=cfg["math_max_new_tokens"],
                    temperature=cfg["generation"]["temperature"],
                    top_p=cfg["generation"]["top_p"],
                    do_sample=cfg["generation"]["do_sample"],
                ),
                "question": row["question"],
                "gold_solution": row["gold_solution"],
                "gold_final": row["gold_final"],
            }
    
    judge = PairwiseAIJudge(cfg, Path(cfg["results_dir"]))
    for policy_name in responses.keys():
        win_rate = 0.
        total_vr = 0.
        format_complied = 0.
        mean_res_len = 0.
        all_lengths = []
        total = 0
        num_agreed = 0
        num_disagreed = 0
        for idx in responses[policy_name]:
            total += 1
            current = responses[policy_name][idx]
            
            vr = exact_reward(current["generation"]["responses"][0], current["gold_final"])
            total_vr += vr
            format_complied += float(extract_designated_final(current["generation"]["responses"][0]) is not None)
            
            mean_res_len += mean_response_length(current["generation"]["response_mask"])
            all_lengths.extend(current["generation"]["response_lengths"])
            
            if policy_name == "sft": continue
            
            baseline = responses["sft"][idx]
            # 0.5 = tie, 1.0 = won, 0.0 = lost
            pairwise_reward = judge.group_rewards(
                current["question"],
                [
                    current["generation"]["responses"][0],
                    baseline["generation"]["responses"][0],
                ]
            )[0]
            win_rate += pairwise_reward
            
            vr_baseline = exact_reward(baseline["generation"]["responses"][0], current["gold_final"])
            # match value with RLAIF
            vr_result = 0.5 if vr_baseline == vr else 1.0 if vr > vr_baseline else 0.0
            num_agreed += int(vr_result == pairwise_reward)
            num_disagreed += int(vr_result != pairwise_reward)
        
        win_rate /= total
        total_vr /= total
        format_complied /= total
        mean_res_len /= total
        std_res_len = torch.std(torch.as_tensor(all_lengths, dtype=torch.float32)).item()
        num_agreed /= total
        num_disagreed /= total
        
        path = Path(cfg["results_dir"]) / "task5_feedback" / f"eval-{policy_name}-{args.dataset}.json"
        path.unlink(missing_ok=True)
        save_json(
            path,
            {
                "win_rate": win_rate,  # not interpret for SFT
                "mean_exact_reward": total_vr,
                "mean_length": mean_res_len,
                "std_length": std_res_len,
                "verifier-judge-agreement": num_agreed,  # not interpret for SFT
                "verifier-judge-disagreement": num_disagreed,  # not interpret for SFT
            }
        )


if __name__ == "__main__":
    main()
