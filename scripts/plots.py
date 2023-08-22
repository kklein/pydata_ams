from functools import cache
from pathlib import Path

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from git_root import git_root


def _root() -> Path:
    return Path(git_root())


def _plot_root() -> Path:
    return _root() / "plots"


@cache
def _df_risotto() -> pd.DataFrame:
    return pd.read_csv(_root() / "data" / "risotto.csv")


# Taken from
# https://matplotlib.org/stable/gallery/statistics/customized_violin.html#violin-plot-customization
def _set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel("Sample name")


def plot_dgp_dag() -> None:
    """Generate directed acyclic graph describing data generating process."""
    dot = graphviz.Digraph("dgp", comment="Data Generating Process", format="png")
    # TODO: To make a node/edge invisible use `style="invis"`.
    dot.node("Age", "Age")
    dot.node("Nationality", "Nationality")
    dot.node("Chef_rating", "Chef rating")
    dot.node("Gaz_stove", "Gaz stove")
    dot.node("Stirring", "Stirring")
    dot.node("Pleasure", "Pleasure")
    dot.edge("Chef_rating", "Gaz_stove")
    for covariate in ["Age", "Nationality", "Chef_rating", "Gaz_stove"]:
        dot.edge(covariate, "Pleasure")
        dot.edge(covariate, "Stirring")
    dot.edge("Stirring", "Pleasure")
    dot.render(directory="plots").replace("\\", "/")


def plot_why_prediction_fails() -> None:
    """Plot violin plot indicating when prediction might fail."""
    df = _df_risotto()
    gas_stoves = df["gas_stove"].unique()
    data = [
        df[df["gas_stove"] == gas_stove]["outcome"].values for gas_stove in gas_stoves
    ]

    fig, ax = plt.subplots()
    ax.violinplot(dataset=data)
    _set_axis_style(ax, gas_stoves)
    ax.set_xlabel("gas stove")
    ax.set_ylabel("pleasure")
    fig.savefig(_plot_root() / "why_prediction_fails.png")


plot_dgp_dag()
plot_why_prediction_fails()
