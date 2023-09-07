---
title: Marp slide deck
description: An example slide deck created by Marp CLI
author: Yuki Hattori
keywords: marp,marp-cli,slide
url: https://marp.app/
image: https://marp.app/og-image.jpg
---

# Causal Inference Libraries: What They Do and What I'd Like Them to Do

---

# Agenda

1. Why Causal Inference and why heterogeneity?
2. How can we estimate heterogeneous treatment effects on paper?
3. How can we estimate heterogeneous treatment effects in practice?
4. What are we missing from EconML and CausalML?

---

# Risotto

- Can either be prepared
  - in a laborous and delicate fashion, involving a lot of **stirring**
  - in a cut-throat, cantine style fashion, **not** involving a lot of
    **stirring**
- Consumers of risotto are **free to decide how much they pay** for
  their risotto.
- Naturally we wonder: should we be stirring?

---

# When prediction fails

What happens when we intervene on a data point from the left,
i.e. `stirring = 0`, and now - keeping everything else unchanged - make sure that a gas stove is used, i.e. `stirring = 1`?

![bg right 100%](../plots/why_prediction_fails_1.png)

---

![bg 50%](../plots/prediction_success.gv.png)
![bg 50%](../plots/prediction_failure.gv.png)

---

![bg 100%](../plots/why_prediction_fails_3.png)
![bg 100%](../plots/why_prediction_fails_2.png)

---

# Why heterogeneity

---

![bg 70%](../plots/treatment_effects_1.png)

---

## To stir or not to stir, the maths

- Assume that the cost of stirring amounts to 1$ per unit.
- Also assume that the overall revenue when never stirring is $R$.
- Then, the overall revenue when **always stirring** is $R - n \cdot 1 +
  \delta_1$
  - The plot from the previous slide tells us that $n \cdot 1 > \delta$.
- The overrall revenue of **stirring when we expect it to pay off**: $R - k \cdot 1 +
  \delta_{\pi}$
  - We can condition on certain 'covariates'/features to decide for whom it
    pays off.
  - When doing this 'right', we get that $\delta_{\pi} > k \cdot 1$.

<!-- TODO: Use colors. -->

---

![bg 70%](../plots/treatment_effects_2.png)

<!-- TODO: Show delta_pi on plot. -->

---

# Estimating heterogeneity on paper

---

## The fundamental problem of Causal Inference

Desire

| Consumer | Age of consumer | Non-stirred outcome/payment | Stirred outcome/payment | Individual treatment effect |
| -------- | --------------- | --------------------------- | ----------------------- | --------------------------- |
| Bob      | 28              | 21                          | 21.8                    | .8                          |
| Anne     | 10              | 12                          | 12                      | 0                           |

---

## The fundamental problem of Causal Inference

Reality

| Consumer | Age of consumer | Non-stirred outcome/payment | Stirred outcome/payment | Individual treatment effect |
| -------- | --------------- | --------------------------- | ----------------------- | --------------------------- |
| Bob      | 28              | 21                          | ?                       | ?                           |
| Anne     | 10              | ?                           | 12                      | ?                           |

---

## What now?

- We can't estimate the Individual Treatment Effect (ITE).
- Yet, we can define an estimand, the Conditional Average Treatment Effect
  (CATE), which we can actually estimate:
  $\tau(X) := \mathbb{E}[\text{payment}|X\text{, stirring}] -
  \mathbb{E}[\text{payment}|X\text{, no stirring}]$

---

## Conventional assumptions for estimating heterogeneous treatment effects

- Positivity/overlap
- Conditional ignorability/unconfoundedness
- SUTVA

A randomized control trial usually gives us the first two for free.

---

## The DR-Learner

---

# Estimating heterogeneity in practice

---

## Risotto consumption: a simulation

![bg left 70%](../plots/dgp.gv.png)

---

## Risotto consumption: a simulation

|    age | nationality | chef_rating | gas_stove | $\mu$ | $T$ | $\tau$ |   $Y$ |
| -----: | :---------- | ----------: | --------: | ----: | --: | -----: | ----: |
|  50.77 | Indonesia   |        0.53 |         1 | 20.73 |   1 |   0.34 | 21.08 |
|  59.48 | Iraq        |        0.46 |         0 | 20.46 |   0 |   0.76 | 20.46 |
|  47.25 | India       |        0.46 |         0 | 24.29 |   0 |   0.19 | 24.29 |
|  22.21 | Italy       |        0.58 |         0 | 15.90 |   1 |   0.88 | 16.79 |
| 100.40 | India       |        0.58 |         1 | 29.95 |   1 |   0.30 | 30.25 |

$\mu(X) \equiv$ the 'base outcome', i.e. outcome/payment without stirring
$T \equiv$ the treatment, whether the risotto has been stirred or not
$\tau(X) \equiv$ the heterogeneous treatment effect
$Y \equiv$ the outcome, the final payment
$Y = \mu(X) + T \cdot \tau(X)$

<!-- TODO: Stress which columns wouldn't usually be available in a -->
<!-- non-simulated context -->

---

```python
# One-hot encoding
X = pd.concat([
    df[numerical_covariates],
    pd.get_dummies(df["nationality"])
], axis=1)

# Model definition
reg = lgbm.LGBMRegressor()
clf = lgbm.LGBMClassifier()
model = causalml.BaseRRegressor(
    outcome_learner=reg,
    effect_learner=reg,
    propensity_learner=clf,
)

# Model training and prediction
model.fit(X=X, treatment=df[treatment], y=df[outcome])
cate_estimates = model.predict(X)
```

---

![bg 70%](../plots/cate_estimates.png)

---

# Problems in practice

---

## 1. Categorical features

- `lightgbm` is a very popular choice for prediction on tabular
  datasets.
- One of is nice features is that it works natively with categorical
  features.
  E.g. instead of having to one-hot encode categoricals, one can
  'tell' `lightgbm` that a single columns is to be treated as a
  categorical.

---

![bg left 150%](../plots/numerical_tree.png)
![bg right 150%](../plots/categorical_tree.png)

---

- Option 1: Use `pandas` `category` dtype

  ```python
  df["nationality"] = df["nationality"].astype("category")
  model = lgbm.LGBMRegressor()
  model.fit(df[["nationality"]], df["payment"])
  ```

- Option 2: Explicitly set `categorical_indices`
  ```python
  df["nationality"] = df["nationality"].astype("category").cat.codes
  model = lgbm.LGBMRegressor(categorical_feature=[0])
  model.fit(df[["nationality"]], df["payment"])
  ```

---

- Unfortunately, both options don't work with `causalml` and `econml`.
- Option 1 is not possible since both convert `pandas` input to `numpy`
  objects in a 'validation' step.
- Option 2 is not possible since constructor parameters can't be
  passed.
- A hack is - of course - possible to indirectly use option 2:
  ```python
  from functools import partialmethod
  from lightgbm import LGBMRegressor
  LGBMRegressor.fit = partialmethod(
    LGBMRegressor.fit,
    categorical_feature=[0],
  )
  ```

---

## Tying back to our example: what's the difference?

![](../plots/one-hot-vs-categorical.png)

---

## 2. Reusing component models

Deciding between meta learners is a hard challenge [insert reference]

- In practice, this often boils down to 'trying out' different ones.
- Many of the meta learners require a component model estimating $\mathbb{E}[Y|X,T=0]$.
- Being able to reuse said model would often easily lead to a halving
  of training cost.

[Insert illustration]

---

## 3. Transparent cross-fitting

---

## 4. Biased final stage

---

## 5. Stage-specific covariate sets
