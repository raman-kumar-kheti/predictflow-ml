# PredictFlow ML (`production-ml-pipeline`)

> **End-to-end machine learning and MLOps platform** for customer churn prediction, experiment tracking, model versioning, drift monitoring, and API serving.

## 1. Problem Statement
Customer churn directly impacts recurring revenue in subscription businesses. Detecting high-risk customers early enables retention teams to intervene before cancellation. This project provides an end-to-end, CPU-compatible production ML pipeline that validates customer data, engineers domain features, trains and compares multiple classifiers with class-imbalance handling, tracks experiments in MLflow, monitors data drift, and serves predictions via FastAPI and a Streamlit dashboard.

## 2. Project Overview
This repository transforms the production engineering patterns from [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML) into a practical **Customer Churn Prediction Platform** built on Scikit-learn, MLflow, FastAPI, Streamlit, Docker, and GitHub Actions.

### Upstream vs. Original Contributions
- **Inherited from `Made-With-ML` (MIT License):** Modular package layout, environment-backed configuration, structured logging, reproducible seed management, MLflow experiment tracking pattern, and automated testing discipline.
- **Original Additions:**
  - Tabular Telco Customer Churn ingestion, schema/quality validation, and cleaning (`TotalCharges` coercion and imputation).
  - Automated EDA report and Matplotlib/Seaborn visualizations (`churn_distribution.png`, `numeric_distributions.png`, `correlation_heatmap.png`, `categorical_churn_rates.png`).
  - Domain feature engineering (`avg_monthly_spend`, `charges_ratio`, `num_services`, `is_long_term_contract`, `has_auto_payment`, `tenure_group`) and `ColumnTransformer` preprocessing.
  - Multi-model training & comparison (Logistic Regression, Random Forest, Gradient Boosting) with `class_weight="balanced"` and `SMOTE` imbalance handling.
  - MLflow SQLite tracking, Model Registry versioning, and automated best-model selection.
  - FastAPI prediction service (`/health`, `/model/info`, `/predict`, `/predict/batch`, `/drift/check`) with confidence scores and risk tiers (`LOW`, `MEDIUM`, `HIGH`).
  - PSI + Kolmogorov-Smirnov data drift monitoring.
  - Streamlit MLOps comparison & interactive drift dashboard.
  - Docker containerization and self-contained GitHub Actions CI.

## 3. Features
- Dataset ingestion, schema validation, and automated missing-value handling
- Automated Exploratory Data Analysis (EDA) with saved plots and JSON summary
- Feature engineering and Scikit-learn `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
- Class-imbalance handling (`class_weight="balanced"` and `SMOTE`)
- Model comparison across Logistic Regression, Random Forest, and Gradient Boosting
- Evaluation with Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix, and Feature Importance
- MLflow experiment tracking, Model Registry versioning, and best-model selection
- FastAPI service for single-customer and batch predictions with confidence scores and model version metadata
- Population Stability Index (PSI) and KS-test data drift monitoring
- Streamlit comparison and drift dashboard
- Docker support, Pytest suite, and GitHub Actions CI

## 4. Technology Stack
- **ML & Data:** Python 3.12, Pandas, NumPy, Scikit-learn, imbalanced-learn, SciPy, Joblib
- **Visualization:** Matplotlib, Seaborn
- **Experiment Tracking & Registry:** MLflow (SQLite backend)
- **API & UI:** FastAPI, Pydantic, Uvicorn, Streamlit
- **Testing & DevOps:** Pytest, HTTPX, Docker, GitHub Actions

## 5. Architecture & Data Flow
1. **Ingestion & Validation (`src/churn_ml/data_ingestion.py`):** Loads `data/raw/telco_customer_churn.csv`, validates required columns/types/target values, and cleans `TotalCharges`.
2. **EDA (`src/churn_ml/eda.py`):** Generates summary statistics and 4 diagnostic plots in `reports/eda/`.
3. **Feature Engineering (`src/churn_ml/features.py`):** Computes churn domain features via `ChurnFeatureEngineer` and scales/encodes via `ColumnTransformer`.
4. **Training & MLflow (`src/churn_ml/train.py`, `src/churn_ml/evaluate.py`):** Trains Logistic Regression, Random Forest, and Gradient Boosting on stratified 80/20 splits, logs metrics/plots/models to MLflow, registers the best model, and saves `artifacts/best_model.joblib` + `artifacts/reference_data.parquet`.
5. **Serving & Monitoring (`api/main.py`, `src/churn_ml/predict.py`, `src/churn_ml/drift.py`):** Serves predictions with confidence scores and checks incoming batches for feature drift.

## 6. Folder Structure
```text
production-ml-pipeline/
├── .github/workflows/ci.yml
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
├── api/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
├── dashboard/
│   └── app.py
├── data/
│   └── raw/telco_customer_churn.csv
├── src/
│   └── churn_ml/
│       ├── __init__.py
│       ├── config.py
│       ├── data_ingestion.py
│       ├── drift.py
│       ├── eda.py
│       ├── evaluate.py
│       ├── features.py
│       ├── predict.py
│       └── train.py
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_data_and_features.py
    └── test_training_and_drift.py
```

## 7. Dataset & License
- **Dataset:** Telco Customer Churn (`7,043` rows, `21` columns)
- **Source:** [IBM Telco Customer Churn Sample Data](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/data/Telco-Customer-Churn.csv) (Apache-2.0 repository)

## 8. Verified Experimental Results
Measured on the 20% stratified holdout split (`1,409` samples, `random_state=42`) on CPU:

| Model | Imbalance Strategy | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `logistic_regression` | `class_weight=balanced` | 0.7331 | 0.4983 | 0.7914 | 0.6116 | 0.8423 |
| `random_forest` | `class_weight=balanced` | 0.7566 | 0.5285 | 0.7674 | 0.6260 | 0.8401 |
| `gradient_boosting` *(Selected)* | `SMOTE` | 0.7757 | 0.5678 | 0.6497 | 0.6060 | 0.8440 |

## 9. Installation & Environment Configuration
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 10. How to Run Locally
```bash
# 1. Run EDA
PYTHONPATH=src python -m churn_ml.eda

# 2. Train models and register best model in MLflow
PYTHONPATH=src python -m churn_ml.train

# 3. Start FastAPI service
PYTHONPATH=src:. uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Start Streamlit dashboard (optional)
PYTHONPATH=src:. streamlit run dashboard/app.py
```

## 11. How to Run Tests
```bash
PYTHONPATH=src:. pytest tests/ -v
```

## 12. API Documentation
Once running, interactive OpenAPI docs are available at `http://localhost:8000/docs`:
- `GET /health` — Service and model load status
- `GET /model/info` — Active model version, MLflow run ID, metrics, and feature importances
- `POST /predict` — Single-customer churn prediction
- `POST /predict/batch` — Batch customer churn predictions
- `POST /drift/check` — PSI and KS-test drift report against training reference data

### Example Input (`POST /predict`)
```json
{
  "customerID": "CUST-0001",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 268.5
}
```

### Example Output
```json
{
  "customer_id": "CUST-0001",
  "churn_prediction": "Yes",
  "churn_probability": 0.7712,
  "confidence_score": 0.7712,
  "risk_tier": "HIGH",
  "model_name": "gradient_boosting",
  "model_version": "1.0.0 (mlflow-v1)"
}
```

## 13. Docker Instructions
```bash
docker build -t production-ml-pipeline .
docker run -p 8000:8000 production-ml-pipeline
```

## 14. Limitations & Future Improvements
- **Limitations:** Trained on a static 7,043-row telco dataset; uses a fixed 0.50 decision threshold.
- **Future Improvements:** Add threshold optimization for business cost curves, scheduled retraining triggers on drift alerts, and Prometheus metrics export.

## 15. Original Repository Attribution & License
- **Original Upstream Repository:** [GokuMohandas/Made-With-ML](https://github.com/GokuMohandas/Made-With-ML)
- **License:** MIT License (see [LICENSE](LICENSE), `Copyright (c) 2023 Made With ML`).
