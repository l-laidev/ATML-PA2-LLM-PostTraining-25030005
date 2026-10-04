from __future__ import annotations

from dotenv import load_dotenv

from common.generation import batch_generate, response_sequence_logprobs, score_reward_pairs
from common.logging_utils import save_json
from common.metrics import mean_response_length, preference_accuracy, sampled_kl
load_dotenv()

import argparse

import torch
import tqdm

from common.data import encode_prompt_response, load_yaml, pad_batch, read_jsonl, repo_path
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["dpo_standard_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)

    def custom_collate(rows, tokenizer):
        chosen, rejected = make_collate(tokenizer, int(bundle["cfg"]["max_sequence_length"]))(rows)
        return [r["prompt"] for r in rows], chosen, rejected

    reward_model, reward_tok = bundle["reward"][0], bundle["reward"][1]
    loader = torch.utils.data.DataLoader(
        bundle["rows"],
        batch_size=int(bundle["cfg"]["batch_size"]),
        shuffle=False,
        collate_fn=lambda rows: custom_collate(rows, bundle["tokenizer"]),
    )
    
    loss = 0.
    KL_ref = 0.
    pref_acc = 0.
    total = 0
    mean_resp_len = 0
    lengths = []
    batched_prompts = []
    batched_responses = []
    with torch.no_grad():
        with torch.inference_mode():
            for prompt, chosen, rejected in tqdm.tqdm(loader, desc="Evaluating", total=len(loader)):
                c = {k: v.to(bundle["policy"].device) for k, v in chosen.items()}
                r = {k: v.to(bundle["policy"].device) for k, v in rejected.items()}

                with reference_mode(bundle["policy"]):
                    ref_chosen, _, _ = response_sequence_logprobs(bundle["policy"], c)
                    ref_rejected, _, _ = response_sequence_logprobs(bundle["policy"], r)

                policy_chosen, _, _ = response_sequence_logprobs(bundle["policy"], c)
                policy_rejected, _, _ = response_sequence_logprobs(bundle["policy"], r)

                cur_loss, _ = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, bundle["cfg"]["beta"])
                cur_pref_acc = preference_accuracy(policy_chosen, policy_rejected)
                # TODO: check KL and log-proba
                cur_KL_ref = 1/2 * (
                      sampled_kl(policy_chosen, ref_chosen, torch.ones_like(policy_chosen))
                    + sampled_kl(policy_rejected, ref_rejected, torch.ones_like(policy_rejected))
                )

                generation = batch_generate(
                    bundle["policy"],
                    bundle["tokenizer"],
                    prompt,
                    bundle["cfg"]["max_sequence_length"],
                    bundle["cfg"]["max_generation_tokens"],
                    bundle["cfg"]["generation"]["temperature"],
                    bundle["cfg"]["generation"]["top_p"],
                    bundle["cfg"]["generation"]["do_sample"],
                )

                batched_prompts.append(list(map(lambda prompt: {"role": "user", "content": prompt}, prompt)))
                batched_responses.append(bundle["tokenizer"].batch_decode(generation["response_ids"], skip_special_tokens=True))
                mean_resp_len += mean_response_length(generation["response_mask"])
                lengths.extend(generation["response_lengths"])

                loss += cur_loss.item()
                pref_acc += cur_pref_acc
                KL_ref += cur_KL_ref.item()
                total += 1

            
            score = 0.
            total = 0
            for prompt, response in tqdm.tqdm(zip(batched_prompts, batched_responses), desc="Scoring", total=len(batched_prompts)):
                cur_score = score_reward_pairs(reward_model, reward_tok, prompt, response, bundle["cfg"]["max_sequence_length"])
                score += cur_score.mean().item()
                total += 1

    save_json(
        f"{bundle['cfg']['results_dir']}/{args.name}-evaluation.json",
        {
            "loss": loss / total if total > 0 else 0,
            "preference_accuracy": pref_acc / total if total > 0 else 0,
            "kl_reference": KL_ref / total if total > 0 else 0,
            "reward_score": score / total if total > 0 else 0,
            "mean_response_length": mean_resp_len / total if total > 0 else 0,
            "response_length_std": float(torch.std(torch.tensor(lengths, dtype=torch.float)).item()),
        }
    )

if __name__ == "__main__":
    main()
