from __future__ import annotations

import argparse
from pathlib import Path
import torch
from tqdm import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import _response_mask, batch_generate, response_token_logprobs
from common.logging_utils import append_jsonl
from common.models import load_tokenizer
from task2_ppo.continue_train import prepare_ppo_continuation, run_ppo
from task2_ppo.ppo import compute_gae, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized

def get_new_logp(bundle, prompt, response, terminated_with_eos):
    tokenizer = bundle["tokenizer"]
    model = bundle["policy"]

    rendered = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_enc = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=bundle["cfg"]["max_prompt_length"],
    )

    device = next(model.parameters()).device
    prompt_ids = prompt_enc["input_ids"].to(device)
    prompt_attention = prompt_enc["attention_mask"].to(device)

    response_enc = tokenizer(
        response,
        return_tensors="pt",
        add_special_tokens=False,
    )
    response_ids = response_enc["input_ids"].to(device)

    
    if terminated_with_eos and (response_ids.shape[1] == 0 or response_ids[0, -1].item() != tokenizer.eos_token_id):
        eos = torch.tensor(
            [[tokenizer.eos_token_id]],
            dtype=response_ids.dtype,
            device=device,
        )
        response_ids = torch.cat([response_ids, eos], dim=1)
    rmask = _response_mask(response_ids, tokenizer.eos_token_id).to(device)


    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.cat([prompt_attention, torch.ones_like(response_ids)], dim=1)

    new_logp, _ = response_token_logprobs(
        model,
        input_ids,
        attention_mask,
        prompt_ids.shape[1],
        response_ids,
    )

    return new_logp, rmask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Required epsilon values:", cfg["clip_values"])
    print("Cache keys:", sorted(rows[0].keys()))

    bundle = prepare_ppo_continuation(args.config)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    device = next(bundle["policy"].parameters()).device

    geometric_results_path = Path(f"{cfg['results_dir']}/geometric-clipping-study.json")
    geometric_results_path.unlink(missing_ok=True)

    for eps in (t := tqdm(cfg["clip_values"])):
        t.set_description_str(f"Clip Eps {eps:.4f}")

        avg_loss = 0.
        avg_frac = 0.
        for row in tqdm(rows, desc="Measuring", leave=False):
            response = row["response"]
            
            prompt = None
            for p in prompts:
                if p["prompt_id"] == row["prompt_id"] and p["source_index"] == row["source_index"]:
                    assert prompt is None
                    prompt = p["messages"]
                    # prompt = p["prompt"]

            new_logp, mask = get_new_logp(bundle, prompt, response, row["terminated_with_eos"])
            
            # add batch dim
            ref_logp = row["ref_logprobs"].unsqueeze(0).to(device)
            old_logp = row["old_logprobs"].unsqueeze(0).to(device)
            values = row["values"].unsqueeze(0).to(device)

            # penalized reward based on terminated_with_eos
            effective_task_reward = torch.tensor(
                row["effective_terminal_reward"],
                device=device,
                dtype=ref_logp.dtype).unsqueeze(0)

            rollout_rewards = shaped_rewards(
                effective_task_reward,
                old_logp,
                ref_logp,
                mask,
                cfg["kl_beta"],
            )

            advantages, returns = compute_gae(
                rollout_rewards,
                values,
                mask,
                cfg["gamma"],
                cfg["gae_lambda"],
            )

            loss, ratio, frac = ppo_policy_loss(new_logp, old_logp, advantages, mask, eps=eps)
            avg_loss += loss.item()
            avg_frac += frac.item()

        append_jsonl(
            geometric_results_path,
            {
                "eps": eps,
                "loss": avg_loss / len(rows),
                "frac": avg_frac / len(rows),
            }
        )

    out = repo_path(cfg["output"])
    out = out / "clipping-study"
    for eps in (t := tqdm(cfg["clip_values"])):
        t.set_description_str(f"Clip Eps {eps:.4f}")
        run_ppo(
            args.config,
            out / f"clip-{eps:.4f}",
            cfg["fork_updates"],
            eps,
            cfg["kl_beta"],
            f"clip-{eps:.4f}",
            token_budget=1703,  # based on the standard PPO run's 8th update
        )


if __name__ == "__main__":
    main()
