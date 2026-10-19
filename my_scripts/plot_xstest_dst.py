from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

from common.data import repo_path
from common.logging_utils import load_json
from my_scripts.common import set_neurips_style


if __name__ == "__main__":
    save_base = repo_path("figures")
    base = repo_path("results") / "task4_safety"
    rates = {}
    for policy in ["sft", "dpo", "ppo", "grpo"]:
        path = base / f"safety_results_{policy}.json"
        rate = load_json(path)
        # transpose so metrics are columns and categories are rows/x-axis
        rates[policy] = pd.DataFrame(rate["category_level"]).T

        rates[policy].index = rates[policy].index.str.replace("_", " ").str.title()
        rates[policy].columns = rates[policy].columns.str.replace("_", " ").str.title()

    set_neurips_style()
    f, ax = plt.subplots(2, 2, figsize=(2 * 3.25, 2 * 2.7), layout="constrained")
    ax = ax.ravel()

    for i, policy in enumerate(rates):
        rate: pd.DataFrame = rates[policy]
        rate = rate.sort_values(by="Justified Refusal Rate")
        
        rate.plot(
            kind="bar", 
            stacked=True, 
            ax=ax[i], 
            width=0.8,
            # colormap="Set2",
            color={
                "Justified Refusal Rate": "#4D6BFE",
                "Safe Answer Rate": "#64748B",
                "Unsafe Compliance Rate": "#1F1F1F",
            },
        )
        
        ax[i].set_title(policy.upper())
        ax[i].set_xlabel("")
        ax[i].tick_params(axis='x', rotation=90)
        
        if ax[i].get_legend():
            ax[i].get_legend().remove()

    handles, labels = ax[0].get_legend_handles_labels()
    f.legend(
        handles, labels, 
        loc="upper center", 
        bbox_to_anchor=(0.5, 1.04), 
        ncols=3, 
        frameon=True
    )

    plt.savefig(save_base / "judge-label-dst.pdf", bbox_inches="tight")