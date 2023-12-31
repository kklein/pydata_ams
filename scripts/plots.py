from copy import deepcopy
from functools import cache, partialmethod
from pathlib import Path

import data_generation
import graphviz
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from causalml.inference.meta import BaseRRegressor
from econml.dr import DRLearner
from git_root import git_root
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error

_TEST_FRACTION = 0.2
_NUMERICAL_COVARIATES = ["age", "chef_rating", "gas_stove"]


def _root() -> Path:
    return Path(git_root())


def _plot_root() -> Path:
    return _root() / "imgs"


@cache
def _df_risotto() -> pd.DataFrame:
    df = pd.read_csv(_root() / "data" / "risotto.csv")
    df["nationality"] = df["nationality"].astype("category")
    return df


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
        format="svg",
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
        format="svg",
        directory=_plot_root(),
    )
    dot_success.node("stirring", "Stirring")
    dot_success.node("payment", "Payment")
    dot_success.edge("stirring", "payment")
    dot_success.render()

    dot_failure = graphviz.Digraph(
        "prediction_failure",
        format="svg",
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
    ax.plot([x0], [y0], "o", markersize=10, c="orange")
    fig.tight_layout()
    fig.savefig(_plot_root() / "why_prediction_fails_1.png")

    fig, ax = deepcopy(_prediction_failure())

    x1, y1 = 2, 4
    # ax.plot([x0], [y0], "o", markersize=10, c="orange")
    ax.plot([x0, x1], [y0, y1], "o", markersize=10, c="orange")
    ax.arrow(
        x0,
        y0,
        (x1 - x0),
        (y1 - y0),
        length_includes_head=True,
        width=0.3,
        head_length=0.3,
        color="orange",
        head_width=1.5,
    )
    fig.tight_layout()
    fig.savefig(_plot_root() / "why_prediction_fails_2.png")

    fig, ax = deepcopy(_prediction_failure())
    x1, y1 = 2, 8
    ax.plot([x0, x1], [y0, y1], "o", markersize=10, c="orange")
    ax.arrow(
        x0,
        y0,
        (x1 - x0),
        (y1 - y0),
        length_includes_head=True,
        width=0.1,
        head_length=0.4,
        color="orange",
        head_width=0.4,
    )
    epsilon = 0.2
    ax.plot([x1 + epsilon, x1 + epsilon], [y0, y1], "-", color="orange")
    mid = (y0 + y1) / 2
    ax.annotate(
        "treatment effect",
        xy=(x1 + epsilon, mid),
        xytext=(x1 + 2 * epsilon, mid),
        arrowprops={"facecolor": "black", "shrink": 0.05},
    )
    fig.tight_layout()
    fig.savefig(_plot_root() / "why_prediction_fails_3.png")


@cache
def _treatment_effect_hist():
    df = _df_risotto()
    fig, ax = plt.subplots()
    n, _, patches = ax.hist(df["treatment_effect"], bins=20)
    ax.set_xlabel("$\\tau$: difference in payment \n (treatment effect)")

    for item in (
        [
            ax.title,
            ax.xaxis.label,
        ]
        + ax.get_xticklabels()
        + ax.get_yticklabels()
    ):
        item.set_fontsize(22)
    return fig, ax, n, patches


def plot_treatment_effects(threshold: int) -> None:
    """Plot treatment effects."""
    fig, ax, _, _ = deepcopy(_treatment_effect_hist())
    fig.tight_layout()
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
        ],
        prop={"size": 22},
    )

    fig.tight_layout()
    fig.savefig(_plot_root() / "treatment_effects_2.png")


def plot_cate_estimates(rng):
    """Plot comparison of cate estimates to actuals."""
    outcome = "payment"
    treatment = "stirring"
    numerical_covariates = ["age", "chef_rating", "gas_stove"]
    df = _df_risotto()

    X = pd.concat([df[numerical_covariates], pd.get_dummies(df["nationality"])], axis=1)

    test_indicator = rng.binomial(n=1, p=0.2, size=len(X))

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
    true_cates = df[test_indicator == 1]["treatment_effect"]
    ax.scatter(cate_estimates_econml, true_cates)
    x_min = min(min(cate_estimates_econml), min(true_cates))
    x_max = max(max(cate_estimates_econml), max(true_cates))
    ax.plot([x_min, x_max], [x_min, x_max], "-")
    fig.tight_layout()

    fig.savefig(_plot_root() / "cate_estimates.png")


def plot_categorical_tree():
    """Plot an arbitrary lgbm tree with a categorical feature."""
    df = _df_risotto()
    model = lgbm.LGBMRegressor(verbosity=-1, max_depth=2)
    model.fit(df[["nationality"]], df["payment"])
    fig, ax = plt.subplots(figsize=(15, 10))
    lgbm.plot_tree(model, ax=ax)
    fig.tight_layout()
    fig.savefig(_plot_root() / "categorical_tree.png", transparent=True)


def plot_numerical_tree():
    """Plot an arbitrary lgbm tree with a numerical feature."""
    df = _df_risotto()
    model = lgbm.LGBMRegressor(verbosity=-1, max_depth=2)
    model.fit(df[["age"]], df["payment"])
    fig, ax = plt.subplots(figsize=(15, 10))
    lgbm.plot_tree(model, ax=ax)
    fig.tight_layout()
    fig.savefig(_plot_root() / "numerical_tree.png", transparent=True)


def _reg_rmse(
    df, test_indicator, treatment: str = "stirring", outcome: str = "payment"
) -> float:
    X = pd.concat(
        [df[_NUMERICAL_COVARIATES], pd.get_dummies(df["nationality"])], axis=1
    )
    X_train, X_test = X.iloc[test_indicator == 0], X.iloc[test_indicator == 1]
    df_train, df_test = df.iloc[test_indicator == 0], df.iloc[test_indicator == 1]
    model_cml = BaseRRegressor(
        outcome_learner=lgbm.LGBMRegressor(verbosity=-1),
        effect_learner=lgbm.LGBMRegressor(verbosity=-1),
        propensity_learner=lgbm.LGBMClassifier(verbosity=-1),
    )
    model_cml.fit(X=X_train, treatment=df_train[treatment], y=df_train[outcome])
    cate_estimates_cml = model_cml.predict(X_test)
    return mean_squared_error(
        cate_estimates_cml, df_test["treatment_effect"], squared=False
    )


def _cat_rmse(
    df, test_indicator, treatment: str = "stirring", outcome: str = "payment"
) -> float:
    df["nationality"] = df["nationality"].astype("category").cat.codes
    df_train, df_test = df.iloc[test_indicator == 0], df.iloc[test_indicator == 1]
    X_train = df_train[_NUMERICAL_COVARIATES + ["nationality"]]
    X_test = df_test[_NUMERICAL_COVARIATES + ["nationality"]]
    HackR = deepcopy(lgbm.LGBMRegressor)
    HackR.fit = partialmethod(
        HackR.fit,
        categorical_feature=[3],
    )
    HackC = deepcopy(lgbm.LGBMClassifier)
    HackC.fit = partialmethod(
        HackC.fit,
        categorical_feature=[3],
    )
    model_cml = BaseRRegressor(
        outcome_learner=HackR(verbosity=-1),
        effect_learner=HackR(verbosity=-1),
        propensity_learner=HackC(verbosity=-1),
    )

    model_cml.fit(X=X_train, treatment=df_train[treatment], y=df_train[outcome])
    cate_estimates_cml = model_cml.predict(X_test)
    return mean_squared_error(
        cate_estimates_cml, df_test["treatment_effect"], squared=False
    )


def plot_categorical_vs_one_hot(
    sample_sizes: tuple[int, ...] = (5000, 10_000, 15_000),
    n_seeds: int = 5,
) -> None:
    """Plot a comparison of R-Learner with categoricals and without."""
    rows = []
    for n in sample_sizes:
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed=seed)
            df = data_generation.gen_covariates(
                n, rng, nations=np.array([f"c{i}" for i in range(500)])
            )
            treatment = data_generation.treatment_assignments(df, rng=rng)
            df_final = data_generation.gen_outcomes(
                df,
                treatment,
                rng=rng,
                mapping={f"c{i}": rng.normal(loc=0.5, scale=2) for i in range(500)},
            )
            test_indicator = rng.binomial(n=1, p=_TEST_FRACTION, size=n)
            rows.append(
                {
                    "type": "one-hot encoding",
                    "n": n,
                    "loss": _reg_rmse(df_final, test_indicator),
                }
            )
            rows.append(
                {
                    "type": "categorical",
                    "n": n,
                    "loss": _cat_rmse(df_final, test_indicator),
                }
            )

    df_stats = pd.DataFrame(rows)
    fig, ax = plt.subplots()
    sns.violinplot(df_stats, hue="type", x="n", y="loss")
    ax.set_ylabel("RMSE")
    ax.set_xlabel("sample size (n)")
    fig.tight_layout()
    fig.savefig(_plot_root() / "one-hot-vs-categorical.png", transparent=True)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=1337)
    plot_dgp_dag()
    plot_why_prediction_fails()
    plot_prediction_failure_dags()
    plot_treatment_effects(threshold=1)
    plot_cate_estimates(rng)
    plot_categorical_tree()
    plot_numerical_tree()
    plot_categorical_vs_one_hot()
