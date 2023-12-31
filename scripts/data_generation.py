from pathlib import Path

import numpy as np
import pandas as pd
from git_root import git_root
from sklearn.preprocessing import MinMaxScaler

NATIONS = np.array(
    ["India", "Italy", "Iceland", "Iraq", "Israel", "Indonesia", "Iran", "Ireland"]
)
MIN_P = 0.1
MAX_P = 0.9
MIN_PRICE = 2.8
MAX_PRICE = 32.2


def gen_covariates(n: int, rng, nations: np.ndarray = NATIONS) -> pd.DataFrame:
    """Generate covariates."""
    # TODO: Consider using scipy's truncnormal distribution instead.
    ages = rng.normal(loc=50, scale=20, size=n)
    ages = np.where(ages < 0, 0, ages)

    nationalities = rng.choice(nations, size=n)

    chef_ratings_raw = rng.normal(loc=6.8, scale=2, size=n)
    # We'd like to apply a logarithm afterwards.
    chef_ratings = (
        MinMaxScaler(feature_range=(0.1, 1))
        .fit_transform(chef_ratings_raw.reshape(-1, 1))
        .flatten()
    )

    gas_stove_scores = chef_ratings + rng.normal(5, scale=3, size=n)
    gas_stove_probabilities = (
        MinMaxScaler(feature_range=(MIN_P, MAX_P))
        .fit_transform(gas_stove_scores.reshape(-1, 1))
        .flatten()
    )
    gas_stoves = rng.binomial(n=1, p=gas_stove_probabilities, size=n)

    df = pd.DataFrame(
        {
            "age": ages,
            "nationality": nationalities,
            "chef_rating": chef_ratings,
            "gas_stove": gas_stoves,
        }
    )
    df["nationality"] = df["nationality"].astype("category")
    df = df.round(decimals=2)
    return df


def _f_p_chef_rating(chef_rating):
    return chef_rating


def treatment_assignments(
    df_covariates: pd.DataFrame, rng, is_rct: bool = False
) -> np.ndarray:
    """Generate treatment assignments for all units."""
    n = len(df_covariates)
    if is_rct:
        return rng.binomial(n=1, p=0.5, size=n)

    score = _f_p_chef_rating(df_covariates["chef_rating"]) + rng.normal(
        loc=0, scale=0.5, size=n
    )
    # We want to ensure positivity and therefore cap the probabilities on both ends.
    normalized_scores = (
        MinMaxScaler(feature_range=(MIN_P, MAX_P))
        .fit_transform(score.to_numpy().reshape(-1, 1))
        .flatten()
    )
    return rng.binomial(n=1, p=normalized_scores, size=n)


def _f_mu_age(age, x_max=50):
    def f_raw(x):
        return x_max**2 - (x - x_max) ** 2

    raws = age.apply(f_raw)
    return MinMaxScaler().fit_transform(raws.to_numpy().reshape(-1, 1)).flatten()


def _f_mu_chef_rating(chef_rating):
    return np.log(chef_rating)


def _f_mu_gas_stove(gas_stove):
    return gas_stove


def _mu(df_covariates, rng) -> np.ndarray:
    n = len(df_covariates)
    score = (
        _f_mu_age(df_covariates["age"])
        + _f_mu_chef_rating(df_covariates["chef_rating"])
        + _f_mu_gas_stove(df_covariates["gas_stove"])
        + rng.normal(loc=0, scale=0.5, size=n)
    )
    return (
        MinMaxScaler(feature_range=(MIN_PRICE, MAX_PRICE))
        .fit_transform(score.to_numpy().reshape(-1, 1))
        .flatten()
    )


def _f_tau_nationality(nationality, rng, mapping=None):
    mapping = mapping or {
        "India": 0.8,
        "Italy": 2,
        "Iceland": 0.3,
        "Iraq": 1.5,
        "Israel": 0.7,
        "Indonesia": 1.1,
        "Iran": 1.2,
        "Ireland": 0,
    }
    result = nationality.apply(lambda x: mapping[str(x)])
    return result.astype("float")


def _tau(
    df_covariates: pd.DataFrame,
    rng,
    mapping=None,
) -> np.ndarray:
    n = len(df_covariates)
    tau_nationality = _f_tau_nationality(
        df_covariates["nationality"], rng, mapping=mapping
    )
    noise = rng.normal(0, 0.1, size=n)
    return tau_nationality + noise


def gen_outcomes(
    df_covariates: pd.DataFrame,
    treatment: np.ndarray,
    rng,
    mapping=None,
) -> pd.DataFrame:
    """Generate outcomes."""
    mu = _mu(df_covariates, rng)
    tau = _tau(df_covariates, rng=rng, mapping=mapping)

    outcome = mu + treatment * tau

    df_outcomes = pd.DataFrame(
        {
            "mu": mu,
            "stirring": treatment,
            "treatment_effect": tau,
            "payment": outcome,
        }
    )

    return pd.concat([df_covariates, df_outcomes], axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(seed=1337)
    df = gen_covariates(20_000, rng=rng)
    treatment = treatment_assignments(df, rng=rng)
    df_final = gen_outcomes(df, treatment, rng=rng)
    df_final.to_csv(Path(git_root()) / "data" / "risotto.csv")
