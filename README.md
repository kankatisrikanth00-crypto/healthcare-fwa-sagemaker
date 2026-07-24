# Healthcare Fraud, Waste & Abuse (FWA) Detection
### End-to-End Machine Learning Pipeline on AWS

---

## Project Overview

Healthcare fraud costs the US healthcare system an estimated **$60 billion per year**. This project builds a complete machine learning pipeline to detect fraudulent healthcare providers using Medicare claims data — the same data structure used by real healthcare payers.

The system analyzes **558,211 claims** across inpatient and outpatient settings, engineers **25 fraud-signal features** at the provider level, and produces a **risk score (0–1)** for each of **5,410 providers** — enabling investigators to prioritize the highest-risk cases.

---

## Results

| Metric | Value |
|--------|-------|
| Providers Scored | 5,410 |
| Claims Analyzed | 558,211 |
| Features Engineered | 25 |
| Dataset Fraud Rate | 9.4% |
| **AUC-PR (headline metric)** | **0.731** |
| Recall | 80% |
| Precision @ Top 50 Flagged Providers | **88%** |
| Top 10 Flagged Providers | **10 out of 10 actual fraud** |

> **Why AUC-PR and not accuracy?** With only 9.4% fraud rate, accuracy is misleading — a model that flags nobody scores 90.6% accuracy. AUC-PR measures performance specifically on the minority fraud class, which is what investigators care about.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                         │
│   Kaggle: Healthcare Provider Fraud Detection Dataset       │
│   4 files: providers.csv, beneficiaries.csv,                │
│            inpatient_claims.csv, outpatient_claims.csv      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     AMAZON S3                               │
│   Bucket: srikanth-fwa-detection                            │
│                                                             │
│   raw/                ← Original CSVs uploaded here        │
│   staging/            ← Intermediate processed files       │
│   curated/            ← Feature tables                     │
│   model-artifacts/    ← Model, scores, plots               │
│     ├── data/train.csv                                      │
│     ├── data/test.csv                                       │
│     ├── xgboost_fraud_model.pkl                             │
│     ├── provider_risk_scores.csv                            │
│     ├── shap_summary.png                                    │
│     └── precision_recall_curve.png                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              AMAZON SAGEMAKER STUDIO                        │
│   Instance: ml.t3.medium (JupyterLab notebook)             │
│                                                             │
│   Step 1 — Data Ingestion                                   │
│     • Load 4 CSVs into pandas                               │
│     • Upload raw files to S3                                │
│                                                             │
│   Step 2 — Feature Engineering (25 features)               │
│     • Combine 558,211 inpatient + outpatient claims         │
│     • Aggregate to provider level                           │
│     • Join beneficiary chronic condition data               │
│     • Attach fraud labels                                   │
│                                                             │
│   Step 3 — Model Training                                   │
│     • Train/test split (80/20, stratified)                  │
│     • XGBoost classifier                                    │
│     • scale_pos_weight=10 to handle class imbalance         │
│     • Early stopping on AUC-PR                              │
│                                                             │
│   Step 4 — Evaluation & Explainability                      │
│     • AUC-PR, Precision-Recall curve                        │
│     • Precision @ Top-K providers                           │
│     • SHAP feature importance                               │
│                                                             │
│   Step 5 — Output                                           │
│     • Risk scores for all 5,410 providers → S3             │
│     • Model artifact → S3                                   │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              AMAZON REDSHIFT SERVERLESS                     │
│   Workgroup: fulfillment-ops-wg                             │
│   Database:  fwa                                            │
│                                                             │
│   Schemas created:                                          │
│     • raw_data   ← source tables                            │
│     • staging    ← intermediate transforms                  │
│     • marts      ← final feature tables for ML             │
│                                                             │
│   Note: Feature engineering performed in SageMaker          │
│   Studio using pandas due to IAM permission constraints.    │
│   In production, dbt on Redshift would run these SQL        │
│   transformations at scale.                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

**Source:** [Healthcare Provider Fraud Detection Analysis — Kaggle](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

| File | Description | Rows |
|------|-------------|------|
| `providers.csv` | Provider ID + fraud label (Yes/No) | 5,410 |
| `beneficiaries.csv` | Patient demographics + chronic conditions | 138,556 |
| `inpatient_claims.csv` | Hospital stay billing records | 40,474 |
| `outpatient_claims.csv` | Doctor visit billing records | 517,737 |

**What these files mean in plain English:**
- A **provider** is a doctor, hospital, or clinic that submits claims to Medicare
- A **beneficiary** is a Medicare patient
- **Inpatient claims** are for hospital stays (high value, longer duration)
- **Outpatient claims** are for doctor visits and procedures (high volume)
- The fraud label (`PotentialFraud = Yes`) means the provider was flagged as fraudulent

---

## Feature Engineering

All features are computed at the **provider level** — one row per provider, summarizing their behavior across all claims.

| Feature | Description | Fraud Signal |
|---------|-------------|--------------|
| `total_claims` | Total number of claims submitted | Very high volume = suspicious |
| `inpatient_claims` | Count of inpatient claims | |
| `outpatient_claims` | Count of outpatient claims | |
| `total_reimbursed` | Total $ reimbursed by Medicare | Extremely high totals = suspicious |
| `avg_claim_amount` | Average $ per claim | Inflated avg = upcoding |
| `max_claim_amount` | Highest single claim | Outlier claims = suspicious |
| `stddev_claim_amount` | Variance in claim amounts | Low variance = templated billing |
| `unique_beneficiaries` | Distinct patients seen | Too many patients = impossible workload |
| `unique_attending_physicians` | Distinct attending doctors | Ghost billing patterns |
| `unique_operating_physicians` | Distinct operating doctors | |
| `avg_deductible_paid` | Average deductible per claim | Waiving deductibles = kickback signal |
| `total_deductible_paid` | Total deductibles collected | |
| `claims_per_beneficiary` | Claims per patient | High ratio = over-billing |
| `inpatient_ratio` | % of claims that are inpatient | Unusually high = upcoding |
| `pct_ChronicCond_*` (11 features) | % of patients with each chronic condition | Targeting vulnerable patients |

---

## AWS Infrastructure

### S3 Bucket Structure
```
s3://srikanth-fwa-detection/
  ├── raw/
  │   ├── providers.csv
  │   ├── beneficiaries.csv
  │   ├── inpatient_claims.csv
  │   └── outpatient_claims.csv
  ├── staging/
  ├── curated/
  └── model-artifacts/
      ├── data/
      │   ├── train.csv          (4,328 rows, 405 fraud)
      │   └── test.csv           (1,082 rows, 101 fraud)
      ├── xgboost_fraud_model.pkl
      ├── provider_risk_scores.csv
      ├── shap_summary.png
      └── precision_recall_curve.png
```

### SageMaker Studio
- **Domain:** admin-project-009846316315
- **Instance type:** ml.t3.medium (JupyterLab)
- **Notebook:** `fwa_detection.ipynb`
- **Libraries:** xgboost 2.1.4, scikit-learn, shap, pandas, boto3

### Redshift Serverless
- **Workgroup:** fulfillment-ops-wg
- **Endpoint:** fulfillment-ops-wg.009846316315.us-east-1.redshift-serverless.amazonaws.com
- **Database:** fwa
- **Schemas:** raw_data, staging, marts
- **Connection method:** Redshift Data API (IAM-based, no password)

---

## Model Details

### Algorithm: XGBoost (Gradient Boosted Trees)

**Why XGBoost for fraud detection?**
- Handles class imbalance via `scale_pos_weight`
- Built-in feature importance
- Fast training on tabular data
- Compatible with SHAP for explainability

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `objective` | binary:logistic | Binary fraud classification |
| `eval_metric` | aucpr | Optimizes for imbalanced classes |
| `scale_pos_weight` | 10 | Compensates for 9.4% fraud rate |
| `max_depth` | 6 | Controls tree complexity |
| `eta` | 0.1 | Learning rate |
| `n_estimators` | 200 | Number of trees |
| `early_stopping_rounds` | 20 | Stops when AUC-PR stops improving |

### Train/Test Split
- **Train:** 4,328 providers (405 fraud, 3,923 clean)
- **Test:** 1,082 providers (101 fraud, 981 clean)
- **Stratified split** to maintain fraud ratio in both sets

### Training Progress
```
[0]    validation_0-aucpr: 0.65454
[20]   validation_0-aucpr: 0.70442
[40]   validation_0-aucpr: 0.72139
[60]   validation_0-aucpr: 0.72096
[70]   validation_0-aucpr: 0.71719  ← early stopping
```

---

## Evaluation

### Classification Report
```
              precision    recall    f1-score   support
           0     0.98       0.91      0.94       981
           1     0.48       0.80      0.60       101
    accuracy                          0.90      1082
```

### Key Metrics Explained

**AUC-PR: 0.731**
The area under the Precision-Recall curve. A random model scores ~0.094 (the fraud base rate). Scoring 0.731 means the model is 7.8x better than random at identifying fraud.

**Precision @ Top 50: 88%**
Of the 50 providers the model flags with the highest risk scores, 44 are actually fraudulent. This is the most operationally important metric — it tells investigators how much time they'll waste on false leads.

**Recall: 80%**
The model catches 80 out of every 100 fraud cases. The remaining 20% are missed — acceptable given we catch 80% while reviewing only a fraction of all providers.

### Top 10 Riskiest Providers
```
Provider    Risk Score    Actual Fraud
PRV54350      0.9947          ✅
PRV52985      0.9942          ✅
PRV51940      0.9939          ✅
PRV51948      0.9939          ✅
PRV55462      0.9938          ✅
PRV52340      0.9938          ✅
PRV56560      0.9938          ✅
PRV55215      0.9937          ✅
PRV56748      0.9936          ✅
PRV52019      0.9936          ✅
```
**10 out of 10 top-flagged providers are actual fraud cases.**

---

## Key Visuals

### SHAP Feature Importance
Shows which features drive the model's fraud predictions.
![SHAP Summary](shap_summary.png)

### Precision-Recall Curve
Shows the trade-off between catching more fraud (recall) vs. reducing false alarms (precision).
![PR Curve](precision_recall_curve.png)

---

## How to Reproduce

### Prerequisites
- AWS account with SageMaker Studio access
- S3 bucket created
- Kaggle account to download dataset

### Steps

**1. Download dataset**
```
Kaggle → "Healthcare Provider Fraud Detection Analysis" by Rohit Rox
Download all 4 CSV files
```

**2. Set up S3**
```
Create bucket: your-fwa-bucket
Create folders: raw/ staging/ curated/ model-artifacts/
Upload 4 CSVs to raw/
```

**3. Open SageMaker Studio**
```
AWS Console → SageMaker → Studio → JupyterLab
Upload 4 CSV files to the notebook environment
Open fwa_detection.ipynb
```

**4. Install dependencies**
```python
!pip install xgboost scikit-learn shap matplotlib pandas boto3 -q
```

**5. Run all cells in order**
```
Cell 1: Install libraries
Cell 2: Load CSVs + upload to S3
Cell 3: Feature engineering
Cell 4: Train/test split
Cell 5: Train XGBoost model
Cell 6: Evaluate + generate plots
Cell 7: Save model + risk scores to S3
```

---

## Production Considerations

In a real healthcare payer environment this pipeline would be extended with:

- **dbt on Redshift** for SQL-based feature engineering with version control and data tests
- **SageMaker Pipelines** to orchestrate ingestion → training → scoring as a scheduled workflow
- **SageMaker Model Monitor** to detect when fraud patterns shift and trigger retraining
- **Human-in-the-loop review** — model flags providers, investigators review before action
- **HIPAA compliance** — PHI encryption at rest (S3 SSE) and in transit (TLS), audit logging via CloudTrail
- **Retraining cadence** — monthly retraining as new fraud labels become available from investigators

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Cloud | AWS (S3, SageMaker Studio, Redshift Serverless) |
| ML Framework | XGBoost 2.1.4 |
| Explainability | SHAP |
| Data Processing | Python, pandas, numpy |
| ML Utilities | scikit-learn |
| AWS SDK | boto3 |
| Dataset | Kaggle — Healthcare Provider Fraud Detection |

---

## Author

**Srikanth Kankati**
Data Engineer & BI Engineer | AWS Certified Solutions Architect
[LinkedIn](https://linkedin.com/in/srikanth-kankati) | [GitHub](https://github.com/srikanth-kankati)
