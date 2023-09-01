from copy import deepcopy
from functools import cache
from pathlib import Path

import graphviz
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dr import DRLearner
from git_root import git_root
from matplotlib.patches import Patch

RNG = np.random.default_rng(seed=1337)


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
    dot.node("payment", "Payment")
    dot.edge("chef_rating", "gas_stove")
    for covariate in ["age", "nationality", "chef_rating", "gas_stove"]:
        dot.edge(covariate, "payment")
        dot.edge(covariate, "stirring")
    dot.edge("stirring", "payment")
    dot.render()


def plot_prediction_failure_dags() -> None:
    """Plot dags indicating why prediction might fail."""
    dot_success = graphviz.Digraph(
        "prediction_success",
        format="png",
        directory=_plot_root(),
    )
    dot_success.node("stirring", "Stirring")
    dot_success.node("payment", "Payment")
    dot_success.edge("stirring", "payment")
    dot_success.render()

    dot_failure = graphviz.Digraph(
        "prediction_failure",
        format="png",
        directory=_plot_root(),
    )
    dot_failure.node("stirring", "Stirring")
    dot_failure.node("payment", "Payment")
    dot_failure.node("chef_rating", "Chef rating")
    dot_failure.edge("chef_rating", "payment")
    dot_failure.edge("chef_rating", "stirring")
    dot_failure.render()


@cache
def _prediction_failure():
    df = _df_risotto()
    gas_stoves = df["stirring"].unique()
    gas_stoves.sort()
    data = [
        df[df["stirring"] == gas_stove]["payment"].values for gas_stove in gas_stoves
    ]

    fig, ax = plt.subplots()
    ax.violinplot(dataset=data)
    _set_axis_style(ax, gas_stoves)
    ax.set_xlabel("stirring")
    ax.set_ylabel("payment")
    return fig, ax


def plot_why_prediction_fails() -> None:
    """Plot violin plot indicating when prediction might fail."""
    x0, y0 = 1, 4
    fig, ax = deepcopy(_prediction_failure())
    ax.plot([x0], [y0], "o")
    fig.savefig(_plot_root() / "why_prediction_fails_1.png")

    fig, ax = deepcopy(_prediction_failure())

    x1, y1 = 2, 4
    ax.plot([x0, x1], [y0, y1], "o")
    ax.arrow(x0, y0, (x1 - x0), (y1 - y0), length_includes_head=True, head_width=0.2)
    fig.savefig(_plot_root() / "why_prediction_fails_2.png")

    fig, ax = deepcopy(_prediction_failure())
    x1, y1 = 2, 7
    ax.plot([x0, x1], [y0, y1], "o")
    ax.arrow(x0, y0, (x1 - x0), (y1 - y0), length_includes_head=True, head_width=0.2)
    epsilon = 0.2
    ax.plot([x1 + epsilon, x1 + epsilon], [y0, y1], "--")
    mid = (y0 + y1) / 2
    ax.annotate(
        "treatment effect",
        xy=(x1 + epsilon, mid),
        xytext=(x1 + 2 * epsilon, mid),
        arrowprops={"facecolor": "black", "shrink": 0.05},
    )
    fig.savefig(_plot_root() / "why_prediction_fails_3.png")


def _treatment_effect_hist():
    df = _df_risotto()
    fig, ax = plt.subplots()
    n, _, patches = ax.hist(df["treatment_effect"], bins=20)
    ax.set_xlabel("treatment effect")
    return fig, ax, n, patches


def plot_treatment_effects(threshold: int) -> None:
    """Plot treatment effects."""
    fig, ax, _, _ = deepcopy(_treatment_effect_hist())
    fig.savefig(_plot_root() / "treatment_effects_1.png")

    fig, ax, n, patches = deepcopy(_treatment_effect_hist())
    for i in range(len(n)):
        x_left = patches[i].get_x()
        x_right = x_left + patches[i].get_width()
        if x_right < 1:
            patches[i].set_facecolor("red")
        elif x_left >= 1:
            patches[i].set_facecolor("green")
        else:
            patches[i].set_facecolor("grey")

    ax.axvline(1, color="orange")
    ax.legend(
        handles=[
            Patch(facecolor="green", edgecolor="green", label="profitable"),
            Patch(facecolor="red", edgecolor="red", label="loss-making"),
            Patch(facecolor="grey", edgecolor="grey", label="on the edge"),
        ]
    )

    fig.savefig(_plot_root() / "treatment_effects_2.png")


def plot_cate_estimates():
    """Plot comparison of cate estimates to actuals."""
    outcome = "payment"
    treatment = "stirring"
    numerical_covariates = ["age", "chef_rating", "gas_stove"]
    df = _df_risotto()

    X = pd.concat([df[numerical_covariates], pd.get_dummies(df["nationality"])], axis=1)

    test_indicator = RNG.binomial(n=1, p=0.2, size=len(X))

    est = DRLearner(
        model_propensity=lgbm.LGBMClassifier(verbosity=-1),
        model_regression=lgbm.LGBMRegressor(verbosity=-1),
        model_final=lgbm.LGBMRegressor(verbosity=-1, num_leaves=2),
    )
    est.fit(
        df[test_indicator == 0][outcome],
        df[test_indicator == 0][treatment],
        X=X[test_indicator == 0],
    )
    cate_estimates_econml = est.effect(X[test_indicator == 1])
    fig, ax = plt.subplots()
    ax.set_xlabel("CATE estimate")
    ax.set_ylabel("CATE")
    ax.scatter(cate_estimates_econml, df[test_indicator == 1]["treatment_effect"])

    fig.savefig(_plot_root() / "cate_estimates.png")


plot_dgp_dag()
plot_why_prediction_fails()
plot_prediction_failure_dags()
plot_treatment_effects(threshold=1)
plot_cate_estimates()
