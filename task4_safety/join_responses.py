

import argparse

import pandas as pd
import tqdm

from common.data import load_yaml, read_jsonl, repo_path
from task4_safety.generate_responses import policy_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    base = repo_path(cfg["results_dir"]) / "task4_safety"
    manual_path = base / "manual_audit_ids.csv"
    manual_df = pd.read_csv(manual_path)
    for policy in policy_specs(cfg).keys():
        print(f"Joining: {policy}")
        path = base / f"generated_{policy}.jsonl"
        generations = read_jsonl(path)
        
        responses = {"xstest_id": [], "policy": [], "benchmark_class": [], "type": [], "prompt": [], "response": []}
        for record in tqdm.tqdm(generations, desc="Extracting responses"):
            responses["xstest_id"].append(record["xstest_id"])
            responses["policy"].append(record["policy"])
            responses["benchmark_class"].append(record["benchmark_class"])
            responses["type"].append(record["type"])
            responses["prompt"].append(record["prompt"])
            responses["response"].append(record["response"])

        responses = pd.DataFrame(responses)
        pd.merge(manual_df, responses, how="inner", on="xstest_id").to_csv(base / f"manual_audit_{policy}.csv")



if __name__ == "__main__":
    main()