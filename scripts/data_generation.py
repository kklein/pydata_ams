from pathlib import Path

import numpy as np
import pandas as pd
from git_root import git_root
from sklearn.preprocessing import MinMaxScaler

RNG = np.random.default_rng(seed=1337)
NATIONS = np.array(["India", "Italy", "Iceland", "Iraq", "Israel", "Indonesia"])
MIN_P = 0.1
MAX_P = 0.9


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
    """Generate treatment assignments for all units."""
    n = len(df_covariates)
    if is_rct:
        return RNG.binomial(n=1, p=0.5, size=n)

    score = _f_p_chef_rating(df["chef_rating"]) + RNG.normal(loc=0, scale=0.5, size=n)
    # We want to ensure positivity and therefore cap the probabilities on both ends.
    normalized_scores = (
        MinMaxScaler(feature_range=(MIN_P, MAX_P))
        .fit_transform(score.to_numpy().reshape(-1, 1))
        .flatten()
    )
    return RNG.binomial(n=1, p=normalized_scores, size=n)


def _f_mu_age(age, x_max=50):
    def f_raw(x):
        return x_max**2 - (x - x_max) ** 2

    raws = age.apply(f_raw)
    return MinMaxScaler().fit_transform(raws.to_numpy().reshape(-1, 1)).flatten()


def _f_mu_chef_rating(chef_rating):
    return np.log(chef_rating)


def _f_mu_gas_stove(gas_stove):
    return gas_stove


def _mu(df_covariates) -> np.ndarray:
    n = len(df_covariates)
    return (
        _f_mu_age(df_covariates["age"])
        + _f_mu_chef_rating(df_covariates["chef_rating"])
        + _f_mu_gas_stove(df_covariates["gas_stove"])
        + RNG.normal(loc=0, scale=0.5, size=n)
    )


def _f_tau_nationality(nationality):
    mapping = {
        "India": 0.4,
        "Italy": 1,
        "Iceland": 0.2,
        "Iraq": 0.5,
        "Israel": 0.5,
        "Indonesia": 0.6,
    }
    return nationality.apply(lambda x: mapping[str(x)])


def _tau(df_covariates) -> np.ndarray:
    n = len(df_covariates)
    return (
        _f_tau_nationality(df["nationality"])
        # Make sure the variance of the noise is sufficient to produce some negative
        # tratment effects.
        + RNG.normal(0, 0.2, size=n)
    )


def gen_outcomes(df_covariates: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
    """Generate outcomes."""
    mu = _mu(df_covariates)
    tau = _tau(df_covariates)

    outcome = mu + treatment * tau

    df_outcomes = pd.DataFrame(
        {
            "mu": mu,
            # TODO: Should we call the treatment 'stirring'?
            "treatment": treatment,
            "treatment_effect": tau,
            # TODO: Should we call the outcome 'pleasure'?
            "outcome": outcome,
        }
    )

    return pd.concat([df_covariates, df_outcomes], axis=1)


if __name__ == "__main__":
    df = gen_covariates(100)
    treatment = treatment_assignments(df)
    df_final = gen_outcomes(df, treatment)
    df_final.to_csv(Path(git_root()) / "data" / "risotto.csv")
