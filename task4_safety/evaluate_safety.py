from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tqdm
from common.data import load_yaml, read_jsonl, repo_path
from common.logging_utils import save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    
    def _compute_metrics(judgements):
        counts = {}
        for judgement in judgements:
            label = judgement["label"]
            label = f"{label}_rate".lower()
            counts[label] = counts.get(label, 0) + 1
        
        total = len(judgements)
        rates = {k: v / total for k,v in counts.items()}
        return counts, rates
    
    for policy in cfg["policies"].keys():
        base = repo_path(cfg["results_dir"]) / "task4_safety"
        path = base / f"safety_results_{policy}.json"
        path.unlink(missing_ok=True)
        
        generations = read_jsonl(base / f"generated_{policy}.jsonl")
        judgements = read_jsonl(base / f"judged_{policy}.jsonl")
        
        combined = {}
        judgements_for_manual_cmp = {"xstest_id": [], "label": []}
        for i in tqdm.trange(len(generations), desc="Merging"):
            idA = generations[i]["xstest_id"]
            idB = judgements[i]["xstest_id"]
            
            if idA not in combined:
                combined[idA] = generations[i]
            else:
                combined[idA] = {
                    **combined[idA],
                    **generations[i]
                }
            
            if idB not in combined:
                combined[idB] = judgements[i]
            else:
                combined[idB] = {
                    **combined[idB],
                    **judgements[i]
                }
            
            judgements_for_manual_cmp["xstest_id"].append(judgements[i]["xstest_id"])
            judgements_for_manual_cmp["label"].append(judgements[i]["label"])
        
        mean_res_len = 0
        judgements_by_type = {}
        for record in combined.values():
            mean_res_len += record["response_tokens"]
            
            if record["type"] not in judgements_by_type:
                judgements_by_type[record["type"]] = []
            judgements_by_type[record["type"]].append({
                k: record[k]
                for k in ["label", "confidence"]
            })
        
        mean_res_len = mean_res_len / len(combined)
        _, rates = _compute_metrics(judgements)
        counts_by_type = {
            k: _compute_metrics(judgements_by_type[k])[0]
            for k in judgements_by_type.keys()
        }
        
        save_json(
            path,
            {**rates,
            "mean_length": mean_res_len,
            "category_level": counts_by_type}
        )
        
        # qualitative evidence
        evidence = {"prompt": [], "response": [], "xstest_id": []}
        for id in combined:
            data = combined[id]
            evidence["xstest_id"].append(id)
            evidence["prompt"].append(data["prompt"])
            evidence["response"].append(data["response"])
        
        manual_judgements = pd.read_json(base / f"manual_audit_{policy}.json")
        automated_judgements = pd.DataFrame(judgements_for_manual_cmp)
        evidence = pd.DataFrame(evidence)
        merged = pd.merge(manual_judgements, automated_judgements, how="inner", on="xstest_id")
        merged = pd.merge(merged, evidence, how="left", on="xstest_id")
        
        agreement_rate = (merged["label"] == merged["manual_label"]).mean()
        ambig_rate_judge = (merged["label"] == "AMBIGUOUS").mean()
        ambig_rate_manual = (merged["manual_label"] == "AMBIGUOUS").mean()
        
        labels = sorted(set(merged["label"]) | set(merged["manual_label"]))
        confusion = pd.crosstab(index=merged["manual_label"], columns=merged["label"]).reindex(index=labels, columns=labels, fill_value=0)
        
        per_label_agreement = pd.DataFrame({
            "manual_label": merged["manual_label"].value_counts(),
            "label": merged["label"].value_counts(),
            "correct": confusion.to_numpy().diagonal(),
        }).astype(int)
        
        print(confusion)
        print(per_label_agreement)
        
        confusion.to_csv(base / f"agreement_{policy}_confusion.csv")
        per_label_agreement.to_csv(base / f"agreement_{policy}_per_label.csv", index=False)
        (merged[merged["label"] != merged["manual_label"]]).to_csv(base / f"disagreement_{policy}.csv", index=False)
        
        save_json(
            base / f"agreement_{policy}.json",
            {
                "agreement_rate": agreement_rate,
                "ambig_rate_judge": ambig_rate_judge,
                "ambig_rate_manual": ambig_rate_manual,
            }
        )
        


if __name__ == "__main__":
    main()
