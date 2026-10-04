from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, set_seed
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    def step(loss, bundle, acc_steps):
        loss = loss / acc_steps
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(bundle["model"].parameters(), bundle["cfg"]["max_grad_norm"])
        bundle["optimizer"].step()
        bundle["optimizer"].zero_grad()
        return loss.item(), norm

    def log(loss, norm, cur_diagnostics):
        append_jsonl(f"{cfg['results_dir']}/{run_name}-diagnostics.jsonl", {
            "loss": loss,
            "norm": norm,
            "logits_mean": cur_diagnostics["logit_mean"].item(),
            "policy_margin_mean": cur_diagnostics["policy_margin_mean"].item(),
            "preference_accuracy": cur_diagnostics["preference_accuracy"].item(),
        })

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    diagnostics = {
        "loss": [],
        "norm": [],
        "logits_mean": [],
        "policy_margin_mean": [],
        "preference_accuracy": [],
    }
    for epoch in range(cfg["epochs"]):
        bundle["optimizer"].zero_grad()
        loss = 0.0
        acc_steps = 0
        for chosen, rejected in tqdm(bundle["loader"], total=len(bundle["loader"]), desc="Training"):
            acc_steps += 1

            c = {k: v.to(bundle["model"].device) for k, v in chosen.items()}
            r = {k: v.to(bundle["model"].device) for k, v in rejected.items()}

            with torch.no_grad():
                with reference_mode(bundle["model"]):
                    ref_chosen, _, _ = response_sequence_logprobs(bundle["model"], c)
                    ref_rejected, _, _ = response_sequence_logprobs(bundle["model"], r)

            policy_chosen, _, _ = response_sequence_logprobs(bundle["model"], c)
            policy_rejected, _, _ = response_sequence_logprobs(bundle["model"], r)

            cur_loss, cur_diagnostics = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, bundle["beta"])
            loss += cur_loss

            if acc_steps == cfg["grad_accum_steps"]:
                loss, norm = step(loss, bundle, acc_steps)
                acc_steps = 0
                log(loss, norm, cur_diagnostics)
        
        if acc_steps != 0:  # leftover steps
            loss, norm = step(loss, bundle, acc_steps)
            acc_steps = 0
            log(loss, norm, cur_diagnostics)

        if loss < best_loss:
            best_loss = loss
            bundle["model"].save_pretrained(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
