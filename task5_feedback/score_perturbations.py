from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import tqdm

from common.data import load_yaml, read_jsonl
from common.logging_utils import append_jsonl
from task5_feedback.rlvr import exact_reward
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    print("Use exact_reward(...) for RLVR and PairwiseAIJudge(...) for RLAIF.")
    
    PAIRS = [
        ("clean_correct", "corrupt_reasoning_correct_final"),  # reasoning sensitivity
        ("clean_correct", "good_reasoning_wrong_final"),  # outcome sensitivity
        ("clean_correct", "persuasive_filler_correct"),  # filler
        ("clean_correct", "gold_distractor_wrong_final"),  # distractor
    ]
    
    judge = PairwiseAIJudge(cfg, Path(cfg["results_dir"]) / "task5_feedback" / f"eval-diagnostic-judge-cache.json")
    save_path = Path(f"{cfg['results_dir']}") / "task5_feedback" / "perturbations.jsonl"
    save_path.unlink(missing_ok=True)
    for (better, worse) in PAIRS:
        print(f"Controlled-diagnostic: {better} vs {worse}")
        
        vr_rates = {
            "better": 0.,
            "worse": 0.,
            "tie": 0.,
        }
        ai_rates = {
            "better": 0.,
            "worse": 0.,
            "tie": 0.,
        }
        for problemID in tqdm.tqdm(groups, desc="Evaluating"):
            problem = groups[problemID]
            
            vrA = exact_reward(problem[better]["response"], problem[better]["gold_final"])
            vrB = exact_reward(problem[worse]["response"], problem[worse]["gold_final"])
            
            # rewarod for [response A, response B]
            pairwise_reward = judge.group_rewards(problem[better]["question"], [problem[better]["response"], problem[worse]["response"]])
            
            vrPref = "tie" if vrA == vrB else "better" if vrA > vrB else "worse"
            aiPref = "tie" if pairwise_reward[0] == 0.5 else "better" if pairwise_reward[0] == 1.0 else "worse"
            
            vr_rates[vrPref] += 1
            ai_rates[aiPref] += 1
        
        total = len(groups)
        vr_rates = {k: v / total for k,v in vr_rates.items()}
        ai_rates = {k: v / total for k,v in ai_rates.items()}
        
        append_jsonl(
            save_path,
            {
                "better": better,
                "worse": worse,
                "rlvr": vr_rates,
                "rlaif": ai_rates,
            }
        )

if __name__ == "__main__":
    main()
