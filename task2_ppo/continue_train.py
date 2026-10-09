from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
import tqdm

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, set_seed, wall_timer
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", token_budget=None):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    if token_budget is not None:
        cfg["generated_token_budget"] = token_budget

    training_results_path = Path(f"{cfg['results_dir']}/{run_name}.json")
    training_results_path.unlink(missing_ok=True)

    shuffle_generator = torch.Generator().manual_seed(6304)
    timer = wall_timer()
    best_loss = torch.inf
    total_generated_tokens = 0.
    
    torch.cuda.reset_peak_memory_stats()

    for upd in tqdm.trange(cfg["updates"], desc="PPO Updates"):
        shuffled = torch.randperm(len(bundle["prompt_rows"]), generator=shuffle_generator)
        shuffled = shuffled[:cfg["prompts_per_update"]]
        prompts = bundle["prompt_rows"][shuffled]
        
        generation = batch_generate(
            bundle["policy"],
            bundle["tokenizer"],
            prompts,
            cfg["max_prompt_length"],
            cfg["max_response_length"],
            cfg["generation"]["temperature"],
            cfg["generation"]["top_p"],
            cfg["generation"]["do_sample"],
        )

        generated_tokens = int(generation["response_mask"].sum().item())
        if "generated_token_budget" in cfg and total_generated_tokens+generated_tokens > cfg["generated_token_budget"]:
            break
        total_generated_tokens += generated_tokens

        for k in ["sequences", "response_ids", "attention_mask"]:
            generation[k] = generation[k].clone()
        
        with reference_mode(bundle["policy"]), torch.no_grad():
            ref_logp, _ = response_token_logprobs(
                bundle["policy"],
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )

        bundle["policy"].eval()
        with torch.no_grad():
            old_policy_logp, _ = response_token_logprobs(
                bundle["policy"],
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )
        bundle["policy"].train()


        task_rewards = score_reward_pairs(
            bundle["reward_model"],
            bundle["reward_tokenizer"],
            prompts,
            generation["responses"],
            cfg["reward_max_length"],
        )
        
        pt_terminated_with_eos = torch.tensor(generation["terminated_with_eos"], dtype=torch.bool, device=task_rewards.device)
        # following the convention based on the cached rollouts in PPO Task 2
        # penalty is subtracted
        task_rewards -= (cfg["missing_eos_penalty"] * (~pt_terminated_with_eos))
        # (not used) convention used in the reference: https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo
        # task_rewards[~pt_terminated_with_eos] = -1 * cfg["missing_eos_penalty"]

        rollout_rewards = shaped_rewards(
            task_rewards,
            old_policy_logp,
            ref_logp,
            generation["response_mask"],
            cfg["kl_beta"],
        )

        with torch.no_grad():
            values = token_values(
                bundle["value_model"],
                generation["sequences"],
                generation["attention_mask"],
            )
            values = values[:, generation["prompt_width"]-1:-1]

        advantages, returns = compute_gae(
            rollout_rewards,
            values,
            generation["response_mask"],
            cfg["gamma"],
            cfg["gae_lambda"],
        )

        advantages = advantages.detach()
        returns = returns.detach()
        advantages = normalize_advantages(advantages, generation["response_mask"])

        logged_rewards = None
        logged_kl_from_ref = None
        logged_policy_loss = None
        logged_value_loss = None
        logged_total_loss = None
        logged_entropy = None
        logged_policy_norm = None
        logged_value_norm = None
        logged_affected_frac = None
        wall_time = None
        
        for epoch in tqdm.trange(cfg["ppo_epochs"], desc="Epochs", leave=False):
            bundle["policy"].eval()
            new_policy_logp, _ = response_token_logprobs(
                bundle["policy"],
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )
            bundle["policy"].train()
            policy_loss, ratio, affected_frac = ppo_policy_loss(
                new_policy_logp,
                old_policy_logp,
                advantages,
                generation["response_mask"],
                cfg["clip_epsilon"],
            )
            
            new_values = token_values(
                bundle["value_model"],
                generation["sequences"],
                generation["attention_mask"],
            )
            new_values = new_values[:, generation["prompt_width"]-1:-1]
            value_loss = value_mse_loss(new_values, returns, generation["response_mask"])
            
            total_loss = policy_loss\
                + cfg["value_coef"] * value_loss
            
            bundle["policy_optimizer"].zero_grad(set_to_none=True)
            bundle["value_optimizer"].zero_grad(set_to_none=True)
            total_loss.backward()
            policy_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(bundle["policy"]), cfg["max_grad_norm"])
            value_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(bundle["value_model"]), cfg["max_grad_norm"])
            bundle["policy_optimizer"].step()
            bundle["value_optimizer"].step()

            if epoch == cfg["ppo_epochs"]-1:
                with torch.no_grad():
                    # update logp for logging
                    bundle["policy"].eval()
                    new_policy_logp, _ = response_token_logprobs(
                        bundle["policy"],
                        generation["sequences"],
                        generation["attention_mask"],
                        generation["prompt_width"],
                        generation["response_ids"],
                    )
                    bundle["policy"].train()

                    # TODO: shaped reward or task reward?
                    logged_rewards = task_rewards.mean().item()
                    logged_kl_from_ref = sampled_kl(new_policy_logp, ref_logp, generation["response_mask"]).item()
                    logged_policy_loss = policy_loss.item()
                    logged_value_loss = value_loss.item()
                    logged_total_loss = total_loss.item()
                    logged_entropy = sample_entropy(new_policy_logp, generation["response_mask"]).item()
                    logged_policy_norm = policy_norm.item()
                    logged_value_norm = value_norm.item()
                    logged_affected_frac = affected_frac.item()
                    wall_time = timer()

                if logged_total_loss < best_loss:
                    best_loss = logged_total_loss
                    bundle["policy"].save_pretrained(out / f"{run_name}-policy")
                    bundle["value_model"].save_pretrained(out / f"{run_name}-value")

        append_jsonl(
            training_results_path,
            {
                "update": upd,
                "mean_reward": logged_rewards,
                "KL_from_ref": logged_kl_from_ref,
                "policy_loss": logged_policy_loss,
                "value_loss": logged_value_loss,
                "total_loss": logged_total_loss,
                "entropy": logged_entropy,
                "policy_norm": logged_policy_norm,
                "value_norm": logged_value_norm,
                "affected_frac": logged_affected_frac,
                "rollout_mean_length": mean_response_length(generation["response_mask"]),
                "rollout_std_length": float(generation["response_mask"].sum(-1).float().std().item()),
                "wall_time": wall_time,
                "peak_VRAM_GiB": torch.cuda.max_memory_allocated() / (1024**3),
                "update_generated_tokens": generated_tokens,
                "total_generated_tokens": total_generated_tokens,
            }
        )

    bundle["policy"].save_pretrained(out / f"{run_name}-final-policy")
    bundle["value_model"].save_pretrained(out / f"{run_name}-final-value")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name)


if __name__ == "__main__":
    main()
