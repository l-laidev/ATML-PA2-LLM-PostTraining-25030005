from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])
    
    out = repo_path(cfg["output"])
    out = out / "kl-study"
    for kl_beta in cfg["kl_values"]:
        print(f"KL Beta {kl_beta:.4f}")
        run_ppo(
            args.config,
            out / f"kl-beta-{kl_beta:.4f}",
            cfg["fork_updates"],
            cfg["clip_epsilon"],
            kl_beta,
            f"kl-beta-{kl_beta:.4f}",
            token_budget=1703,  # based on the standard PPO run's 8th update
        )


if __name__ == "__main__":
    main()
