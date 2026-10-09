from __future__ import annotations

import argparse
from common.data import load_yaml
from task3_grpo.continue_train import run_grpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    
    run_grpo(
        args.config,
        f"{cfg['output']}/norm-study",
        cfg["fork_updates"],
        loss_type="grpo",
        run_name="type-grpo",
        # TODO: determine the budget
        token_budget=9999,
    )
    
    run_grpo(
        args.config,
        f"{cfg['output']}/norm-study",
        cfg["fork_updates"],
        loss_type="dr_grpo",
        run_name="type-dr-grpo",
        token_budget=9999,
    )


if __name__ == "__main__":
    main()
