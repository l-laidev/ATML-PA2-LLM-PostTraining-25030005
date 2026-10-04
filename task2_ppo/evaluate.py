from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from common.data import load_yaml, read_jsonl
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.metrics import mean_response_length, sample_entropy, sampled_kl
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo.ppo import shaped_rewards


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
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    
    eval_results_path = Path(bundle["cfg"]["results_dir"])
    eval_results_path.parent.mkdir(parents=True, exist_ok=True)

    logged_reward = 0.
    logged_rollout_reward = 0.
    logged_kl_from_ref = 0.
    logged_entropy = 0.
    logged_mean_len = 0.
    lengths = []
    logged_stability = 0.
    n_total = 0
    with torch.no_grad(), torch.inference_mode():
        for row in tqdm(bundle["rows"], desc="Evaluating"):
            generation = batch_generate(
                bundle["policy"],
                bundle["tokenizer"],
                [row["prompt"]],
                bundle["cfg"]["max_prompt_length"],
                bundle["cfg"]["max_response_length"],
                bundle["cfg"]["generation"]["temperature"],
                bundle["cfg"]["generation"]["top_p"],
                bundle["cfg"]["generation"]["do_sample"],
            )

            with reference_mode(bundle["policy"]):
                ref_logp, _ = response_token_logprobs(
                    bundle["policy"],
                    generation["sequences"],
                    generation["attention_mask"],
                    generation["prompt_width"],
                    generation["response_ids"],
                )
            
            bundle["policy"].eval()
            cur_logp, _ = response_token_logprobs(
                bundle["policy"],
                generation["sequences"],
                generation["attention_mask"],
                generation["prompt_width"],
                generation["response_ids"],
            )
            bundle["policy"].train()

            task_rewards = score_reward_pairs(
                bundle["reward"][0],  # model
                bundle["reward"][1],  # tokenizer
                [row["prompt"]],
                generation["responses"],
                bundle["cfg"]["reward_max_length"]
            )
            pt_terminated_with_eos = torch.tensor(generation["terminated_with_eos"], dtype=torch.bool, device=task_rewards.device)
            task_rewards -= (bundle["cfg"]["missing_eos_penalty"] * (~pt_terminated_with_eos))
            
            rollout_rewards = shaped_rewards(
                task_rewards,
                cur_logp,
                ref_logp,
                generation["response_mask"],
                bundle["cfg"]["kl_beta"],
            )

            logged_reward += task_rewards.mean().item()
            logged_rollout_reward += rollout_rewards.mean().item()
            logged_kl_from_ref += sampled_kl(cur_logp, ref_logp, generation["response_mask"]).item()
            logged_entropy += sample_entropy(cur_logp, generation["response_mask"]).item()
            logged_mean_len += mean_response_length(generation["response_mask"])
            lengths.extend(generation["response_lengths"])
            logged_stability += 0.  # TODO: stability stat
            n_total += 1

    save_json(
        f"{bundle['cfg']['results_dir']}/held-out-evaluation/{args.name}",
        {
            "mean_task_reward": logged_reward / n_total,
            "mean_rollout_reward": logged_rollout_reward / n_total,
            "kl_from_ref": logged_kl_from_ref / n_total,
            "entropy": logged_entropy / n_total,
            "mean_response_length": logged_mean_len / n_total,
            "std_response_length": torch.as_tensor(lengths, dtype=torch.float32).std().item(),
            "stability": logged_stability / n_total,
        }
    )


if __name__ == "__main__":
    main()
