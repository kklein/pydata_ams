from pathlib import Path

import numpy as np
import pandas as pd
from git_root import git_root
from sklearn.preprocessing import MinMaxScaler

RNG = np.random.default_rng(seed=1337)
NATIONS = np.array(["India", "Italy", "Iceland", "Iraq", "Israel", "Indonesia"])


def gen_covariates(n: int) -> pd.DataFrame:
    """Generate covariates."""
    # TODO: Consider using scipy's truncnormal distribution instead.
    ages = RNG.normal(loc=50, scale=20, size=n)
    ages = np.where(ages < 0, 0, ages)

    nationalities = RNG.choice(NATIONS, size=n)

    chef_ratings = RNG.normal(loc=6.8, scale=2, size=n)
    # We'd like to apply a logarithm afterwards.
    chef_ratings = np.where(chef_ratings <= 0.1, 0.1, chef_ratings)

    gas_stove_scores = chef_ratings + RNG.normal(5, scale=3, size=n)
    gas_stove_probabilities = (
        MinMaxScaler().fit_transform(gas_stove_scores.reshape(-1, 1)).flatten()
    )
    gas_stoves = np.where(gas_stove_probabilities >= 0.5, 1, 0)

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
    df_covariates: pd.DataFrame, is_rct: bool = False
) -> np.ndarray:
    """Generate treatment assignments for all units based on covariates."""
    n = len(df_covariates)
    if is_rct:
        return RNG.binomial(n=1, p=0.5, size=n)
    score = _f_p_chef_rating(df["chef_rating"]) + RNG.normal(loc=0, scale=0.5, size=n)
    # TODO: This transformation violates positivity - we need to smoothen!
    return MinMaxScaler().fit_transform(score.to_numpy().reshape(-1, 1)).flatten()


def _f_mu_age(age, x_max=50):
    def f_raw(x):
        return x_max**2 - (x - x_max) ** 2

    raws = age.apply(f_raw)
    return MinMaxScaler().fit_transform(raws.to_numpy().reshape(-1, 1)).flatten()


def _f_mu_nationality(nationality):
    mapping = {
        "India": 0.4,
        "Italy": 1,
        "Iceland": 0.2,
        "Iraq": 0.5,
        "Israel": 0.5,
        "Indonesia": 0.6,
    }
    return nationality.apply(lambda x: mapping[str(x)])


def _f_mu_chef_rating(chef_rating):
    return np.log(chef_rating)


def _f_mu_gas_stove(gas_stove):
    return gas_stove


def gen_outcomes(df_covariates: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
    """Generate outcomes."""
    n = len(df_covariates)
    mu = (
        _f_mu_age(df_covariates["age"])
        + _f_mu_nationality(df_covariates["nationality"])
        + _f_mu_chef_rating(df_covariates["chef_rating"])
        + _f_mu_gas_stove(df_covariates["gas_stove"])
        + RNG.normal(loc=0, scale=0.5, size=n)
    )
    treatment_effect = np.zeros(n)

    outcome = mu + treatment * treatment_effect

    df_outcomes = pd.DataFrame(
        {
            "mu": mu,
            "treatment": treatment,
            "treatment_effect": treatment_effect,
            "outcome": outcome,
        }
    )

    return pd.concat([df_covariates, df_outcomes], axis=1)


if __name__ == "__main__":
    df = gen_covariates(100)
    treatment = treatment_assignments(df)
    df_final = gen_outcomes(df, treatment)
    df_final.to_csv(Path(git_root()) / "data" / "risotto.csv")
