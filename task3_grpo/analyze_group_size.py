from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl
from common.logging_utils import append_jsonl


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Students should decide and document exactly how prompts/completions are partitioned for the
    requested equal-generation comparison.
    """
    
    # N x K needs to be fixed for same total-generation budget
    # K=2, then N=24 => 48 total
    # K=4, then N=12 => 48 total
    # K=8, then N=6 => 48 total
    group_ids = list(by_prompt.keys())
    group_budget = len(group_ids) // (k//2)  # k/2 = 1, 2, 4 => N/(k/2) = 24, 12, 6
    rollout_budget = k
    
    rng = np.random.default_rng(seed=6304)
    group_ids = rng.choice(group_ids, size=group_budget, replace=False)
    
    regrouped = {}
    for gid in group_ids:
        rollouts = by_prompt[gid]
        rollouts = rng.choice(rollouts, size=rollout_budget, replace=False)
        regrouped[gid] = rollouts
    return regrouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))
    
    def _compute_metrics(rewards):
        informative_rate = (rewards.std(axis=-1) >= 1e-6).mean()
        mean_within_group_std = rewards.std(axis=-1).mean()
        
        # based on group_relative_advantages function
        mu = rewards.mean(axis=-1, keepdims=True)
        denom = rewards.std(axis=-1, keepdims=True)
        group_rel_var = np.var((rewards - mu) / np.clip(denom, a_min=1e-6, a_max=None))
        
        return {
            "informative_rate": informative_rate.tolist(),
            "mean_within-group_std": mean_within_group_std.tolist(),
            "advantage_variance": group_rel_var.tolist(),
        }
    
    results_path = Path(cfg["results_dir"]) / "group-size.json"
    results_path.unlink(missing_ok=True)
    for k in cfg["group_sizes"]:
        regrouped = regroup_equal_generation_budget(by_prompt, k=k)
        
        rewards = []
        rewards_difficult = []
        rewards_easy = []
        for gid in regrouped:
            rollouts = regrouped[gid]
            rewards_per_group = [r["reward"] for r in rollouts]
            rewards.append(rewards_per_group)
            
            if np.std(rewards_per_group) < 0.1:
                rewards_easy.append(rewards_per_group)
            else:
                rewards_difficult.append(rewards_per_group)
            
        rewards = np.array(rewards)
        rewards_easy = np.array(rewards_easy)
        rewards_difficult = np.array(rewards_difficult)
        
        metrics = _compute_metrics(rewards)
        metrics["group_size"] = k
        if len(rewards_easy) != 0:
            metrics["easy"] = _compute_metrics(rewards_easy)
        if len(rewards_difficult) != 0:
            metrics["difficult"] = _compute_metrics(rewards_difficult)
        
        append_jsonl(
            results_path,
            metrics
        )


if __name__ == "__main__":
    main()
