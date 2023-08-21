import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

rng = np.random.default_rng(seed=1337)
NATIONS = np.array(["Inidia", "Italy", "Iceland", "Iraq", "Isreal", "Indonesia"])


def gen_covariates(n: int) -> pd.DataFrame:
    # TODO: Consider using scipy's truncnormal distribution instead.
    ages = rng.normal(loc=50, scale=20, size=n)
    ages = np.where(ages < 0, 0, ages)

    nationalities = rng.choice(NATIONS, size=n)

    chef_ratings = rng.normal(loc=6.8, scale=2, size=n)

    gas_stove_scores = chef_ratings + rng.normal(5, scale=3, size=n)
    gas_stove_probabilities = (
        MinMaxScaler().fit_transform(gas_stove_scores.reshape(-1, 1)).flatten()
    )
    gas_stoves = np.where(gas_stove_probabilities >= 0.5, 1, 0)

    df = pd.DataFrame(
        {
            "age": ages,
            "nationality": nationalities,
            "chef_reating": chef_ratings,
            "gas_stove": gas_stoves,
        }
    )
    df["nationality"] = df["nationality"].astype("category")
    df = df.round(decimals=2)
    return df


def _f_mu_age(age, x_max=50):
    def f_raw(x):
        return x_max**2 - (x - x_max) ** 2

    raws = age.apply(f_raw)
    return MinMaxScaler().fit_transform(raws.to_numpy().reshape(-1, 1)).flatten()


def _f_mu_nationality(nationality):
    return np.zeros(len(nationality))


def _f_mu_chef_rating(chef_rating):
    return np.zeros(len(chef_rating))


def _f_mu_gas_stove(gas_stove):
    return np.zeros(len(gas_stove))


def gen_outcomes(df_covariates: pd.DataFrame):
    n = len(df_covariates)
    mu = (
        _f_mu_age(df_covariates["age"])
        + _f_mu_nationality(df_covariates["nationality"])
        + _f_mu_chef_rating(df_covariates["chef_rating"])
        + _f_mu_gas_stove(df_covariates["gas_stove"])
        + rng.normal(loc=0, scale=10, size=n)
    )
    treatment_effect = np.zeros(n)

    outcome = mu + treatment_effect

    df_outcomes = pd.DataFrame(
        {
            "mu": mu,
            "treatment_effect": treatment_effect,
            "outcome": outcome,
        }
    )

    return pd.concat([df_covariates, df_outcomes], axis=1)


df = gen_covariates(100)
df_final = gen_outcomes(df)
