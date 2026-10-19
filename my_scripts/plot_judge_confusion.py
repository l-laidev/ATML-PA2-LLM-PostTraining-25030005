from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns

from common.data import repo_path
from my_scripts.common import set_neurips_style


if __name__ == "__main__":
    save_base = repo_path("figures")
    base = repo_path("results") / "task4_safety"
    confusions = {}
    for policy in ["sft", "dpo", "ppo", "grpo"]:
        path = base / f"agreement_{policy}_confusion.csv"
        confusions[policy] = pd.read_csv(path, index_col=0).rename_axis("Manual")

    set_neurips_style()
    s = 1.5
    f, ax = plt.subplots(2,2, figsize=(s*3.25, s*2.7), layout="constrained")
    ax = ax.ravel()
    color_list = ["#FFFFFF", "#64748B", "#4D6BFE"] 
    gradient = LinearSegmentedColormap.from_list("custom_gradient", color_list)

    for i, policy in enumerate(confusions):
        conf: pd.DataFrame = confusions[policy]
        sns.heatmap(
            conf,
            annot=True,
            fmt="d",
            cmap=gradient,
            square=True,
            cbar=False,
            cbar_kws={
                "shrink": 0.8,
                "label": "Instances",
            },
            linewidths=0.75,
            linecolor="#f0f0f0",
            ax=ax[i],
        )

        ax[i].set_title(policy.upper() + " / Judge")
        ax[i].set_ylabel("Human")

        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45//2, ha="right")
        ax[i].set_yticklabels(ax[i].get_yticklabels(), rotation=0)

    plt.savefig(save_base / "confusions.pdf")