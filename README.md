# Predicting Food Production and Nutritional Healthiness in Brazil

Supervised machine learning on the **PHFood Brazil** dataset. The project covers two tasks:

1. **Regression:** predict `Food Production (tons)` for a food item in a given Brazilian state and year.
2. **Classification:** predict whether a food item is `Healthy` (1) or not (0), based on its nutrient profile.

> **Course:** CSE445 (A group of four(4) members )

---

## Dataset

**Planetary Health: A Comprehensive View of Food in Brazil – PHFood Brazil**
Teruel, B. & Miele, M. (2026). Mendeley Data, Version 12. DOI: [10.17632/mt4mj23j73.12](https://data.mendeley.com/datasets/mt4mj23j73/12)
License: CC BY 4.0

The dataset integrates Brazilian open data (IBGE, ANVISA, FAOSTAT, Ministry of Agriculture) covering 1974–2024. It includes harvested area, production, nutrient availability, water use, pesticide residues and more, by food, state and year. Related article: [PLOS ONE, doi:10.1371/journal.pone.0329353](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0329353).

The raw table has **439,299 rows × 52 columns**. It was split into two CSV files (`PHFood_Brazil_Part1.csv`, `PHFood_Brazil_Part2.csv`) and concatenated when loaded. The preprocessing notebook reads them from the project's GitHub repository (`RihalMahmood/CSSE445-Assignments`, `Datasets/` folder). To run locally, download the data from Mendeley and point the loader at your own copy.

---

## Repository Contents

| File | Description |
|---|---|
| `Preprocessing.ipynb` | Cleaning, outlier handling, imputation, label creation, encoding, splitting, scaling, EDA. Writes the `.pkl` files. |
| `Supervised.ipynb` | Loads the `.pkl` files, trains and tunes regression and classification models, prints performance tables. |

Running `Preprocessing.ipynb` produces 12 pickle files (`X_{train,val,test}_{reg,clf}_scaled.pkl` and `y_{train,val,test}_{reg,clf}.pkl`), which `Supervised.ipynb` consumes. **Run the notebooks in that order.**

---

## Pipeline

### 1. Preprocessing (`Preprocessing.ipynb`)

| Step | What happens |
|---|---|
| Load | Concatenate the two CSV parts → 439,299 × 52 |
| Drop missing targets | Remove rows where `Food Production (tons)` is missing → 432,688 rows |
| Drop sparse columns | Remove Water Need/Use/Deficit, Water Deficit %, Retinol, Selenium (6 columns) |
| Drop irrelevant columns | Remove IDs, pesticide/toxicity fields, vitamins, food-group and acquisition columns (27 columns) → 19 columns |
| Type fixes | Convert string scientific-notation values to `float64` |
| Impossible values | Values above hand-set upper thresholds (per column) are set to NaN |
| Outliers | Log-transform (`log1p`) numeric columns and flag IQR outliers (1.5×IQR) in log-space as NaN |
| Deduplicate | `drop_duplicates()` → **52,727 rows** |
| Impute | Median for numeric columns, mode for categorical columns |
| Label | Create the `Healthy` target (see below) |
| Encode | One-hot encode `Food (English)`, `Region`, `State` (`drop='first'`) |
| Split | 70 / 15 / 15 train/val/test, `random_state=42` |
| Scale | `StandardScaler` on numeric columns, fit on train only |

**Final features:** 82 columns (`Year`, `Harvest Area`, 12 nutrient totals, and one-hot columns for food, region and state).
**Split sizes:** 36,908 train / 7,909 validation / 7,910 test.

### Target definitions

- **Regression target:** `Food Production (tons)`
- **Classification target:** `Healthy`, derived from a hand-crafted health score:

```
Health_Score = 2·Protein + 1.5·Fiber
             + 0.5·(Calcium + Iron + Magnesium + Phosphorus + Potassium)
             − 0.5·Carbohydrates − 1·Lipids
Healthy = 1 if Health_Score ≥ median(Health_Score), else 0
```

Note: `Healthy` is a rule-based label. It is not an externally validated measure of nutritional quality. The class balance printed in the notebook is 36,467 healthy vs 16,260 unhealthy.

### 2. Modeling (`Supervised.ipynb`)

| Model | Regression tuning | Classification tuning |
|---|---|---|
| Linear / Logistic Regression | none | Grid Search |
| Random Forest | Grid Search | Random Search |
| XGBoost | Random Search | Bayesian Optimization |
| Neural Network | Manual grid (Keras, early stopping, scaled target) | `MLPClassifier` (100, 50), no tuning |
| K-Nearest Neighbors | Bayesian Optimization | Grid Search |

Cross-validation is 5-fold on the training set. Regression is scored on R², classification on F1.

---

## Results

### Regression: `Food Production (tons)`

| Model | Test RMSE | Test R² | Best params |
|---|---:|---:|---|
| Linear Regression | 5,742,351 | 0.5355 | – |
| Random Forest (Grid) | 650,026 | 0.9940 | depth 10, 200 trees, min split 2 |
| **XGBoost (Random)** | **313,630** | **0.9986** | 200 trees, depth 3, lr 0.3 |
| Neural Network (Grid) | 615,925 | 0.9947 | (100, 50), lr 0.01, batch 64 |
| KNN (Bayesian) | 621,619 | 0.9946 | k=13, p=1, distance-weighted |

### Classification: `Healthy`

| Model | Test Accuracy | Test F1 | Best params |
|---|---:|---:|---|
| Logistic Regression (Grid) | 0.8238 | 0.8780 | C=1, lbfgs |
| **Random Forest (Random)** | **0.9994** | **0.9995** | 100 trees, depth 20, min split 2 |
| XGBoost (Bayesian) | 0.9991 | 0.9994 | lr≈0.152, depth 6, 153 trees |
| Neural Network (MLP) | 0.7994 | 0.8401 | (100, 50) |
| KNN (Grid) | 0.9492 | 0.9641 | k=5, uniform |

Full train/validation/test metrics are in the performance tables at the end of `Supervised.ipynb`.

---

## Limitations

Please read these before citing the numbers above.

- **Label leakage in classification.** `Healthy` is computed directly from nutrient columns (protein, fiber, calcium, etc.), and those same columns are in the feature set. Tree models can recover the scoring rule almost exactly, which explains the ~0.999 accuracy. Treat these scores as "how well can the model reverse-engineer the formula," not as real predictive power. A fairer setup would exclude the score's input columns, or use an independent health label.
- **Near-leakage in regression.** The nutrient columns are aggregates (per-ton nutrient content × production), so they are strongly tied to production itself. The high R² partly reflects that relationship.
- **Preprocessing before splitting.** Outlier removal, median/mode imputation, and the median threshold for `Healthy` were computed on the full dataset before the train/val/test split. The one-hot encoder was also fit on the full dataset. Only the `StandardScaler` is fit on the training split alone. The effect is probably small but nonzero.
- **Large row loss.** The dataset shrinks from 432,688 to 52,727 rows after outlier masking, imputation and `drop_duplicates()`. Setting outliers to NaN and then median-imputing can create many duplicate rows that are then dropped, so the final sample is not a uniform subsample of the original data.
- **Random split on panel data.** Rows are split randomly, not by year or state, so the model sees the same food/state combinations from other years during training. Performance on future years or unseen states is not measured.
- **Convergence warnings.** `LogisticRegression` with `lbfgs` hit the iteration limit; increasing `max_iter` would be a sensible fix.
- **Unused columns.** `Health_Score` is dropped from the feature matrix, but the nutrient columns that define it are kept (see first point).

---

## Setup

Python 3.12 was used.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scikit-optimize tensorflow
```

Then run:

```bash
jupyter notebook Preprocessing.ipynb   # generates the .pkl files
jupyter notebook Supervised.ipynb      # trains and evaluates models
```

The neural network grid search and Bayesian searches are the slowest parts of `Supervised.ipynb`.

---

## Citation

> Teruel, B., & Miele, M. (2026). *Planetary Health: A Comprehensive View of Food in Brazil – PHFood Brazil* (Version 12) [Data set]. Mendeley Data. https://doi.org/10.17632/mt4mj23j73.12

