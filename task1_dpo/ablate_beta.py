from __future__ import annotations

import argparse
from common.data import load_yaml, repo_path
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Required beta values:", cfg["betas"])
    print("Short-run examples per condition:", cfg["short_ablation_examples"])

    out = repo_path(cfg["standard_output"])
    for beta in cfg["betas"]:
        print(f"Launching DPO training for beta={beta}...")
        run_training(args.config, f"beta-{beta}", None, out / f"beta-{beta}", beta, cfg["short_ablation_examples"])

    # raise NotImplementedError(
    #     "TODO(student): launch matched DPO beta forks from the original policy initialization and evaluate them under a common protocol."
    # )


if __name__ == "__main__":
    main()
