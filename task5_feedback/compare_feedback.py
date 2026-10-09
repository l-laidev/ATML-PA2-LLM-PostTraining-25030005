from __future__ import annotations

import argparse
from pathlib import Path

from pandas import read_json
from common.data import load_yaml
from common.logging_utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    base = Path(cfg["results_dir"]) / "task5_feedback"
    
    results = {}
    for policy in ["sft", "rlvr", "rlaif"]:
        results[policy] = {}
        
        for dataset in ["gsm", "transfer"]:
            path = base / f"eval-{policy}-{dataset}.json"
            results[policy][dataset] = read_json(path)
    
    # win rate, exact RLVR reward, mean length, std length, verifier-judge agree/disagreements
    metrics = list(results["sft"]["gsm"].keys())
    drop_results = {}
    for policy in results:
        drop_results[policy] = {}
        
        gsm = results[policy]["gsm"]
        transfer = results[policy]["transfer"]
        
        for m in metrics:
            # SFT doesn't/shouldn't have these
            if policy == "sft" and m in [
                    "win_rate",
                    "verified-judge-agreement",
                    "verified-judge-disagreement",
                ]: continue
            
            rel_change = (transfer[m] - gsm[m]) / gsm[m]
            drop_results[policy][m] = rel_change
    
    save_json(
        base / "perf_drop.json",
        drop_results,
    )

if __name__ == "__main__":
    main()
