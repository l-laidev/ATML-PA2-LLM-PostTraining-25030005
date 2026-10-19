from math import ceil

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from common.data import repo_path
from my_scripts.common import plot_bar, plot_line, set_neurips_style


def plot(logs: pd.DataFrame, metric_to_readable):
    nrows = 3
    f, ax = plt.subplots(nrows,ceil(len(metric_to_readable)/nrows), figsize=(2.0*3.25, 1.5*2.7), layout="constrained")
    ax = ax.ravel()

    _normalize_latex = lambda x: x.replace(r"\sigma", "")  # to show GRPO std response length unrotated
    configs = [
        {
            "column": m,
            "lineplot_kwargs": {"ax": ax[i], "color": "#4D6BFE"},
            "ylabel_kwargs": {"rotation": 0 if len(_normalize_latex(metric_to_readable[m])) < 20 else 90, "va": 'center', "labelpad": 10},
            "title": "",
            "xlabel": r"$\text{Update} \#$",
            "ylabel": metric_to_readable[m],
        } for i, m in enumerate(metric_to_readable)
    ]
    plot_line(logs, configs)

    for unused in range(len(metric_to_readable), len(ax)):
        ax[unused].remove()

    for i, metric in enumerate(metric_to_readable):
        values = logs[metric]
        min_, max_ = values.min(), values.max()
        ax[i].set_yticks(np.arange(min_, max_, (max_ - min_) / 5))
        ax[i].grid()


if __name__ == "__main__":
    config = {
        "ppo": {
            "path": "results/task2_ppo/standard.json",
            "metric_to_readable": {
                "mean_shaped_reward": r"$\mu_{r_{\psi}}$",
                "KL_from_ref": r"$D_{\text{KL}}$",
                "policy_loss": r"$L_{\text{Pol}}$",
                "value_loss": r"$L_{\text{Val}}$",
                "entropy": r"$\text{Entropy} \quad H(\cdot)$",
                "affected_frac": r"$\text{Clip Fraction}$",
                "policy_norm": r"$||\nabla L_\text{Policy}||_2$",
                "value_norm": r"$||\nabla L_\text{Value}||_2$",
                "rollout_mean_length": r"$\mu_{\text{len}}$",
            }
        },
        "grpo": {
            "path": "results/task3_grpo/standard.json",
            "metric_to_readable": {
                "mean_reward": r"$\mu_{r_{\psi}}$",
                "KL": r"$D_{\text{KL}}$",
                "mean_within-group_std": r"$\text{Mean} \quad \sigma_{r_{\psi,g}}$",
                "uninfo_frac": r"$\text{Uninformative}$",
                "loss": r"$L_{\text{GRPO}}$",
                "norm": r"$||\nabla L_\text{GRPO}||_2$",
                "entropy": r"$\text{Entropy} \quad H(\cdot)$",
                "mean_length": r"$\mu_{\text{len}}$",
                "std_length": r"$\sigma_{\text{len}}$",
            }
        },
    }

    base = repo_path("figures")
    base.mkdir(exist_ok=True)

    set_neurips_style()
    all_logs = {}
    for run in config:
        all_logs[run] = pd.read_json(config[run]["path"], lines=True)
        plot(all_logs[run], config[run]["metric_to_readable"])
        plt.savefig(base / f"{run}.pdf")

    f, ax = plt.subplots(1,2, figsize=(3.25, 2.7))
    ax = ax.ravel()
    for i, metric in enumerate(["wall_time", "peak_VRAM_GiB"]):
        values, categories, groups = [], [], []
        for run in all_logs:
            values += [all_logs[run][metric].max()]
            categories += [run.upper()]
            groups += [metric]
        
        unit = "s" if "wall" in metric else "GiB"
        ylabel = "Seconds" if "wall" in metric else "Giga Bytes"
        xlabel = "Wall Time" if "wall" in metric else "Peak VRAM"
        color = ["#8A9AA9", "#4D6BFE"]

        _to_latex_text = lambda x: rf"$\text{{{x}}}$"
        unit = _to_latex_text(unit)
        ylabel = _to_latex_text(ylabel)
        xlabel = _to_latex_text(xlabel)
        categories = list(map(_to_latex_text, categories))

        plot_bar(values, categories, hue=groups, xlabel=xlabel, ylabel=ylabel, fmt=f"%.2f {unit}", figsize=(3.25, 2.7), color=color, fig=f, ax=ax[i])
    plt.savefig(base / f"wall-clock_vram-usage.pdf")