---
title: Marp slide deck
description: An example slide deck created by Marp CLI
author: Yuki Hattori
keywords: marp,marp-cli,slide
url: https://marp.app/
image: https://marp.app/og-image.jpg
---

# Agenda

- 0-5’: Why Causal Inference and why CATE estimation?
- 5-10’: What are some conceptual ways of estimating CATEs?
- 10-20’: How can we use EconML and CausalML for CATE estimation on a real dataset?
- 20-30’: What are we missing from EconML and CausalML?

---

# When prediction fails (?)

![bg right 100%](../plots/why_prediction_fails_1.png)
stress interventional aspect

---

![bg 50%](../plots/prediction_success.gv.png)
![bg 50%](../plots/prediction_failure.gv.png)

---

![bg 100%](../plots/why_prediction_fails_2.png)
![bg 100%](../plots/why_prediction_fails_3.png)

---

- Interventions!
- Jonas Peters example:
  https://www.youtube.com/watch?v=zvrcyqcN9Wo&list=PLzERW_Obpmv-_EXTV1zTmlv-Ab5Tfbp8X
- https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html
- Quote from prediction machines

---

# Why heterogeneity

---

![bg 70%](../plots/treatment_effects_1.png)

---

Simple math: cost of stirring 1USD per unit

- Revenue of never stirring: X
- Revenue of always stirring: X - Y
- Revenue of stirring for those where it pays off: X + Z

---

![bg 70%](../plots/treatment_effects_2.png)

---

# Estimating heterogeneity

---

```
	age	nationality	chef_rating	gas_stove	mu	stirring	treatment_effect	payment
0	50.77	Indonesia	0.53	1	20.737829	1	0.344297	21.082125
1	59.48	Iraq	    0.46	0	20.462730	0	0.760332	20.462730
2 	47.25	India	    0.46	0	24.294204	0	0.194722	24.294204
3  	22.21	Italy	    0.58	0	15.903796	1	0.887621	16.791417
4  	100.40	India	    0.58	1	29.951843	1	0.307483    30.259326
```

---

```py
X = pd.concat([df[numerical_covariates], pd.get_dummies(df["nationality"])], axis=1)

model = BaseRRegressor(
    outcome_learner=lgbm.LGBMRegressor(verbosity=-1, num_leaves=4),
    effect_learner=lgbm.LGBMRegressor(verbosity=-1, num_leaves=4),
    propensity_learner=lgbm.LGBMClassifier(verbosity=-1, num_leaves=4),
)

model.fit(X=X, treatment=df[treatment], y=df[outcome])
model.predict(X)
```

---
