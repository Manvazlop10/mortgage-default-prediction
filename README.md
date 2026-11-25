# Mortgage Default Prediction: 5-Model Classification Comparison

> **Academic project** from a university course.

Predict whether a borrower will default on a mortgage from age, marital status, annual income, mortgage amount, and total amount paid. Two K-Nearest Neighbors models and three decision trees are compared on a held-out test set, and the best model scores 20 new applicants.

**Notebook:** [`notebooks/mortgage_default_prediction.ipynb`](notebooks/mortgage_default_prediction.ipynb)

> **Dataset not included.** The course-provided CSVs are not redistributed, so the notebook is published without outputs. The results below come from the original course run.

## Approach

1. 80/20 train/test split (`random_state=1`): 240 training and 60 test records.
2. Five classifiers:
   - `knn_5`, `knn_10`: KNN with k = 5 and k = 10
   - `mortgage_tree`: entropy decision tree, `min_impurity_decrease=0.03`
   - `mortgage_tree_minsamplesplit`: same, plus `min_samples_split=70`
   - `mortgage_tree_minsampleleaf`: same, plus `min_samples_leaf=50`
3. Compare per-class precision and recall, weighted F1, accuracy, and complexity (k or number of leaves).
4. Score 20 unseen applicants with the selected model.

## Key results (original course run)

| Model | Accuracy | Weighted F1 | Notes |
| --- | ---: | ---: | --- |
| knn_5 | 77% | 0.7656 | Best balance; about 70% recall on defaults |
| knn_10 | 77% | n/a | 91% recall on NO, only 59% on defaults |
| mortgage_tree | 77% | 0.7640 | Baseline tree |
| mortgage_tree_minsampleleaf | 61.7% | n/a | Collapsed to 4 leaves, 37% recall on defaults |

- **Selected model:** `knn_5`.
- **New applicants:** 8 of 20 (40%) flagged as likely to default, mostly younger applicants (24 to 31) with lower income and high mortgage-to-income ratios.

## Limitations and next steps

- Features are not scaled, which biases KNN toward the dollar columns. Next: `StandardScaler` in a scikit-learn `Pipeline`.
- A 60-record test set is small; cross-validation would give more stable estimates.
- In credit risk, missing a default costs more than a false alarm, so class weights or threshold tuning to raise recall on YES are worth testing.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place mortgagedefault.csv and mortgage_prediction.csv in data/
jupyter notebook notebooks/mortgage_default_prediction.ipynb
```

## Tools

Python, pandas, scikit-learn, Matplotlib, Jupyter.

## Author

Manuel Vazquez Lopez
