# Stellar Classification (Galaxy / Quasar / Star)

A 3-class classification project predicting the type of astronomical object —
**GALAXY, QSO (quasar), or STAR** — from SDSS-style photometric measurements.

**Source:** Kaggle Playground Series (Season 6, Episode 6).
*Replace with the exact competition link.*
**Stack:** Python, XGBoost.
**Official metric:** balanced accuracy.

## Approach

- **EDA:** target distribution (imbalanced classes), missing-value check, and a look
  at the survey footprint (`alpha`/`delta` sky coordinates) to confirm they encode
  *where* an object is observed, not *what* it is — and are therefore dropped.
- **Feature engineering:** all preprocessing lives in a **single function applied
  identically to train and test**, guaranteeing the model sees the same
  transformations in both. The key engineered features are the photometric
  **colors** (pairwise magnitude differences across the u, g, r, i, z filters),
  which is where the classes separate physically.
- **Class imbalance:** balanced `sample_weight` is used so the three classes count
  equally, consistent with the balanced-accuracy metric.
- **Model:** XGBoost.
- **Evaluation:** validation balanced accuracy, a confusion matrix (the residual
  confusion is mainly GALAXY ↔ QSO), and feature importances (redshift dominates,
  as expected physically).
- **Submission:** the test set is processed with the *same* preprocessing function,
  guarded by `assert` checks that fail loudly if columns are misaligned.

## Files

- `stellar_class.ipynb` — full notebook (with rendered outputs).

## How to run

Install `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`,
download `train.csv` / `test.csv` from the competition page, and place them next to
the notebook.
