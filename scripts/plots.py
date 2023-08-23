from copy import deepcopy
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
    dot = graphviz.Digraph(
        "dgp",
        comment="Data Generating Process",
        format="png",
        directory=_plot_root(),
    )
    # TODO: To make a node/edge invisible use `style="invis"`.
    dot.node("age", "Age")
    dot.node("nationality", "Nationality")
    dot.node("chef_rating", "Chef rating")
    dot.node("gas_stove", "Gas stove")
    dot.node("stirring", "Stirring")
    dot.node("pleasure", "Pleasure")
    dot.edge("chef_rating", "gas_stove")
    for covariate in ["age", "nationality", "chef_rating", "gas_stove"]:
        dot.edge(covariate, "pleasure")
        dot.edge(covariate, "stirring")
    dot.edge("stirring", "pleasure")
    dot.render()


def plot_prediction_failure_dags() -> None:
    """Plot dags indicating why prediction might fail."""
    dot_success = graphviz.Digraph(
        "prediction_success",
        format="png",
        directory=_plot_root(),
    )
    dot_success.node("gas_stove", "Gas stove")
    dot_success.node("pleasure", "Pleasure")
    dot_success.edge("gas_stove", "pleasure")
    dot_success.render()

    dot_failure = graphviz.Digraph(
        "prediction_failure",
        format="png",
        directory=_plot_root(),
    )
    dot_failure.node("gas_stove", "Gas stove")
    dot_failure.node("pleasure", "Pleasure")
    dot_failure.node("chef_rating", "Chef rating")
    dot_failure.edge("chef_rating", "pleasure")
    dot_failure.edge("chef_rating", "gas_stove")
    dot_failure.render()


@cache
def _prediction_failure():
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
    return fig, ax


def plot_why_prediction_fails() -> None:
    """Plot violin plot indicating when prediction might fail."""
    fig, ax = deepcopy(_prediction_failure())
    fig.savefig(_plot_root() / "why_prediction_fails_1.png")

    fig, ax = deepcopy(_prediction_failure())
    x0, y0 = 1, 2
    x1, y1 = 2, 2
    ax.plot([x0, x1], [y0, y1], "o")
    ax.arrow(x0, y0, (x1 - x0), (y1 - y0), length_includes_head=True, head_width=0.1)
    fig.savefig(_plot_root() / "why_prediction_fails_2.png")

    fig, ax = deepcopy(_prediction_failure())
    x0, y0 = 1, 2
    x1, y1 = 2, 4
    ax.plot([x0, x1], [y0, y1], "o")
    ax.arrow(x0, y0, (x1 - x0), (y1 - y0), length_includes_head=True, head_width=0.1)
    fig.savefig(_plot_root() / "why_prediction_fails_3.png")


plot_dgp_dag()
plot_why_prediction_fails()
plot_prediction_failure_dags()
