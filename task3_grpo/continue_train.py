from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.metrics import mean_response_length
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences
import torch
from torch.optim import AdamW

import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import append_jsonl, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode, trainable_parameters


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo", run_name: str = "standard", token_budget=None):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    
    if token_budget is not None:
        cfg["generated_token_budget"] = token_budget
    
    policy, tokenizer, prompts = cfg["policy"], cfg["tokenizer"], cfg["prompt_rows"]
    rm_model, rm_tokenizer = cfg["reward_model"], cfg["reward_tokenizer"]
    optimizer = cfg["optimizer"]
    
    training_results_path = Path(f"{cfg['results_dir']/{run_name}.json}")
    training_results_path.unlink(missing_ok=True)
    
    shuffle_generator = torch.Generator().manual_seed(6304)
    timer = wall_timer()
    best_loss = torch.inf
    total_generated_tokens = 0.
    
    torch.cuda.reset_peak_memory_stats()
    
    for upd in tqdm.trange(cfg["updates"], desc="GRPO Updates"):
        shuffled = torch.randperm(len(prompts), generator=shuffle_generator)
        shuffled = shuffled[:cfg["prompts_per_update"]]
        group_ids = list(range(shuffled))
        
        # [P1, P2] -> [P1, P1, P2, P2]
        shuffled = torch.repeat_interleave(shuffled, cfg["num_generations"])
        # [0, 1] -> [0, 0, 1, 1]
        group_ids = torch.repeat_interleave(group_ids, cfg["num_generations"])
        
        shuffled_prompts = prompts[shuffled]
        
        generation = batch_generate(
            policy,
            tokenizer,
            shuffled_prompts,
            cfg["max_prompt_length"],
            cfg["max_completion_length"],
            cfg["generation"]["temperature"],
            cfg["generation"]["top_p"],
            cfg["generation"]["do_sample"],
        )
        
        # TODO: should 1 group be counted for each rollout?
        generated_tokens = int(generation["response_mask"].sum().item())
        if "generated_token_budget" in cfg and total_generated_tokens+generated_tokens > cfg["generated_token_budget"]:
            break
        total_generated_tokens += generated_tokens
        
        mask = generation["response_mask"]
        if cfg["mask_truncated_completions"]:
            mask = mask_truncated_sequences(mask, generation["truncated"])
        
        policy.eval()
        with reference_mode(policy), torch.no_grad():
            ref_logp = response_token_logprobs(
                policy,
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )
        
        with torch.no_grad():
            old_logp = response_token_logprobs(
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
            shuffled_prompts,
            generation["responses"],
            max_length=cfg["max_prompt_length"]+cfg["max_completion_length"],
        )
        
        rewards_by_group = rewards.view((cfg["prompts_per_update"], cfg["num_generations"]))

        advantages = group_relative_advantages(rewards, group_ids)
        
        for epoch in tqdm.trange(cfg["policy_epochs"], desc="GRPO Epochs", leave=False):
            policy.eval()
            new_logp = response_token_logprobs(
                policy,
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )
            policy.train()
            
            loss, metrics = grpo_policy_loss(
                new_logp,
                old_logp,
                advantages,
                mask,
                ref_logp,
                cfg["clip_epsilon"],
                cfg["kl_beta"],
                loss_type=loss_type,
                max_completion_length=cfg["max_completion_length"]
            )
            
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), cfg["max_grad_norm"])
            optimizer.step()
            
            if loss < best_loss:
                best_loss = loss
                policy.save_pretrained(f"{cfg['output']}/{run_name}-best")
    
        append_jsonl(
            training_results_path,
            {
                "mean_reward": rewards.mean().item(),
                "KL": metrics["sampled_kl"].item(),
                "mean_within-group_std": rewards_by_group.std(unbiased=False, dim=-1).mean().item(),
                # based on the starter code (1e-6 tolerance)
                "uninfo_frac": ((rewards_by_group.std(unbiased=False, dim=-1) < 1e-6).sum() / cfg["prompts_per_update"]),
                "loss": loss.item(),
                "norm": norm,
                "entropy": metrics["sample_entropy"].item(),
                "mean_length": mean_response_length(generation["response_mask"]),
                "std_length": np.std(generation["response_lengths"]),
                "wall_time": timer(),
                "peak_VRAM_GiB": torch.cuda.max_memory_allocated() / (1024**3),
                "update_generated_tokens": generated_tokens,
                "total_generated_tokens": total_generated_tokens,
            }
        )
    
    policy.save_pretrained(f"{cfg['output']}/{run_name}-final")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()
