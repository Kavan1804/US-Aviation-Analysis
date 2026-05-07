# US Aviation — Flight Delay Severity Analysis

A machine learning project that predicts the **severity** of U.S. domestic flight delays using 3 million flight records. The pipeline covers data cleaning, feature engineering, model training (GPU-accelerated via cuML), explainability, and an interactive Streamlit dashboard.

> **Environment:** The full pipeline was developed and executed on **Google Colab** using an NVIDIA T4 GPU. GPU-accelerated libraries (RAPIDS cuML, cuDF) were used for preprocessing and model training to handle the 3M-row dataset efficiently.

---

## Problem Statement

Rather than predicting whether a flight will be delayed (binary), this project predicts **how severe** a delay will be — a more actionable question for airlines and passengers.

| Class | Label | Definition |
|---|---|---|
| 0 | On-time / Minor | Departure delay ≤ 15 min |
| 1 | Moderate | Departure delay 16–60 min |
| 2 | Severe | Departure delay > 60 min |

Class distribution in the dataset: **82.5% / 11.5% / 6.1%** — heavily imbalanced, handled via oversampling during training.

---

## Dataset

- **Source:** U.S. Bureau of Transportation Statistics (BTS)
- **Size:** ~3 million domestic flight records (`Data/raw/flights_sample_3m.csv`)
- **Features used (19):** airline, origin, destination, origin/destination state, distance, scheduled departure/arrival times, elapsed time, block time, year, month, day, day of week, weekend flag, departure hour, arrival hour, departure time bucket, season

---

## Project Structure

```
US-Aviation-Analysis/
│
├── Code/
│   ├── project_config.py          # Shared paths and directory setup
│   ├── cleaning.py                # 15-step data cleaning pipeline
│   ├── feature_engineering.py     # Delay class labeling + calendar features
│   ├── build_datasets.py          # Builds model-ready CSV
│   ├── train_models.py            # Trains Logistic Regression + Random Forest
│   ├── AI_explainability.py       # Permutation feature importance + local examples
│   ├── build_dashboard_data.py    # RF inference on test set for dashboard
│   └── streamlit_app.py           # Interactive results dashboard
│
├── Data/
│   ├── raw/                       # flights_sample_3m.csv (input)
│   └── processed/                 # cleaned_flights.csv, flights_model_ready_3class.csv
│
├── Models/
│   ├── logistic_regression_3class.joblib
│   └── random_forest_3class.joblib
│
├── Figures/                       # All generated plots (confusion matrices, feature importance, etc.)
├── Reports/                       # JSON reports + CSV metric outputs
│
└── pynb/                          # Jupyter notebook versions (local runs)
    ├── aviation_preprocessing.ipynb
    ├── train_models_.ipynb
    ├── build_dashboard.ipynb
    └── ai_explanatory.ipynb
```

---

## Pipeline

### Option A — Run Locally

```bash
# 1. Clean raw data
python Code/cleaning.py

# 2. Feature engineering + build model-ready dataset
python Code/feature_engineering.py
python Code/build_datasets.py

# 3. Train models (Logistic Regression + Random Forest)
python Code/train_models.py

# 4. Explainability (permutation importance + local prediction examples)
python Code/AI_explainability.py

# 5. Build dashboard prediction dataset
python Code/build_dashboard_data.py

# 6. Launch Streamlit dashboard
streamlit run Code/streamlit_app.py
```

### Option B — Run on Google Colab (GPU-Accelerated)

Colab notebooks are provided as `.md` files in the project root. Each `.md` contains numbered cells to copy into a Colab notebook. **Requires T4 GPU runtime.**

| Notebook | File | What it does |
|---|---|---|
| Preprocessing | `colab_preprocessing.md` | cuDF-accelerated cleaning of 3M rows |
| Model Training | `colab_train_models.md` | cuML Logistic Regression + Random Forest |
| Dashboard Data | `colab_build_dashboard_data.md` | RF inference on test set |
| Explainability | `colab_ai_explainability.md` | Permutation importance + prediction examples |

---

## Models

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.8247 | 0.3013 | 0.7455 |
| Random Forest | 0.5913 | 0.3962 | 0.6548 |

> **Note:** Overall accuracy is misleading here due to class imbalance (82.5% of flights are on-time). **Macro F1 is the primary metric** — it weights all three delay classes equally. Random Forest achieves better minority class recall after balanced training.

---

## Outputs

| File | Description |
|---|---|
| `Reports/cleaning_report.json` | Row counts, missing value stats, Winsorization bounds |
| `Reports/model_training_report.json` | Full metrics for all models |
| `Reports/per_class_metrics.csv` | Precision, recall, F1 per delay class per model |
| `Reports/feature_importance.csv` | Random Forest permutation importance scores |
| `Reports/local_prediction_examples.csv` | High-confidence correct + incorrect predictions |
| `Reports/dashboard_data_report.json` | Summary stats for dashboard predictions |

---

## Dashboard

Run the Streamlit app to explore results interactively:

```bash
streamlit run Code/streamlit_app.py
```

The dashboard has 6 sections:

1. **Overview** — project story, delay class definitions, top-level metrics
2. **Model Performance** — grouped bar chart + interactive confusion matrix
3. **Class-Level Analysis** — per-class precision / recall / F1 with metric toggle
4. **Explainability** — top-N feature importance chart with adjustable slider
5. **Prediction Examples** — correct vs incorrect high-confidence predictions
6. **Takeaways** — summary of findings and limitations

---

## Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib plotly streamlit joblib
```

For GPU-accelerated Colab runs:
```bash
pip install --extra-index-url https://pypi.nvidia.com cuml-cu12 cudf-cu12
```

---

## Key Findings

- Flight delay severity is meaningfully predictable using only **schedule-based features** (no real-time weather or ATC data)
- The strongest predictors are **scheduled departure time, route characteristics, departure hour, and season**
- Severe delays (Class 2, >60 min) are the hardest to predict — they require real-time operational data not available in BTS records
- Macro F1 of ~0.40 with balanced Random Forest vs ~0.30 without — confirming that class imbalance handling is critical for minority delay classes
