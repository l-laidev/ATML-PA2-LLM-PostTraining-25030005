from __future__ import annotations

import argparse
from pathlib import Path

import torch
import tqdm

from common.data import load_yaml, read_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.grpo import group_relative_advantages, mask_truncated_sequences


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    
    results_path = Path(cfg["results_dir"]) / f"{args.name}-heldout-eval.json"
    results_path.unlink(missing_ok=True)
    
    cfg = bundle["cfg"]
    policy, tokenizer = bundle["policy"], bundle["tokenizer"]
    rm_model, rm_tokenizer = bundle["reward"]
    eval_prompts = bundle["rows"]
    
    logged_reward = 0.
    logged_within_group_std = 0.
    logged_num_uninfo = 0
    logged_ent = 0.
    logged_kl = 0.
    logged_mean_len = 0
    total = 0
    
    all_lengths = []
    
    with torch.no_grad(), torch.inference_mode():
        for cur_prompt in tqdm.tqdm(eval_prompts, desc="Evaluating"):
            prompts = [cur_prompt["prompt"] for _ in range(cfg["num_generations"])]
            
            generation = batch_generate(
                policy,
                tokenizer,
                prompts,
                cfg["max_prompt_length"],
                cfg["max_completion_length"],
                cfg["generation"]["temperature"],
                cfg["generation"]["top_p"],
                cfg["generation"]["do_sample"],
            )
            
            mask = generation["response_mask"]
            if cfg["mask_truncated_completions"]:
                mask = mask_truncated_sequences(mask, generation["truncated"])
            
            policy.eval()
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    policy,
                    generation["sequences"],
                    generation["attention_mask"],
                    generation["prompt_width"],
                    generation["response_ids"],
                )
            new_logp, _ = response_token_logprobs(
                policy,
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )
            policy.train()
            
            rewards = score_reward_pairs(
                rm_model,
                rm_tokenizer,
                prompts,
                generation["responses"],
                max_length=cfg["max_prompt_length"]+cfg["max_completion_length"]
            )
            
            logged_reward += rewards.mean().item()
            logged_within_group_std += rewards.std().item()
            logged_num_uninfo += (rewards.std() < 1e-6).item()
            logged_ent += sample_entropy(new_logp, mask).item()
            logged_kl += sampled_kl(new_logp, ref_logp, mask).item()
            logged_mean_len += mean_response_length(generation["response_mask"])
            total += 1
            
            all_lengths.extend(generation["response_lengths"])
    
    # TODO: save examples
    save_json(
        results_path,
        {
            "mean_reward": logged_reward / total,
            "mean_within-group_std": logged_within_group_std / total,
            "frac_uninfo": logged_num_uninfo / total,
            "mean_ent": logged_ent / total,
            "mean_kl": logged_kl / total,
            "mean_len": logged_mean_len / total,
            "std_len": torch.std(torch.as_tensor(all_lengths, dtype=torch.float32)).item(),
        }
    )


if __name__ == "__main__":
    main()
