# Advanced Customer Churn Prediction System

A production-ready machine learning system for predicting customer churn with explainable AI, real-time predictions, and interactive dashboards.


## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)

## Overview

This project implements an **advanced customer churn prediction system** that goes beyond traditional ML projects by incorporating:
- Explainable AI with SHAP values
- Counterfactual explanations for actionable insights
- Real-time prediction capabilities
- Interactive business intelligence dashboard
- ROI-focused business impact analysis

**Business Value:** The system identifies high-risk customers and provides specific, actionable recommendations to prevent churn, potentially saving millions in revenue.

## Key Features

### 1. **Advanced Feature Engineering**
- RFM (Recency, Frequency, Monetary) metrics
- Synthetic temporal/behavioral data generation
- Engagement trend analysis
- 70+ engineered features from 20 base features

### 2. **Multiple ML Models with Experiment Tracking**
- Random Forest (Best: ROC AUC 0.835)
- XGBoost
- LightGBM
- Logistic Regression (Baseline)
- MLflow integration for experiment tracking

### 3. **Explainable AI**
- **SHAP (SHapley Additive exPlanations)**: Global and local feature importance
- **DICE-ML**: Counterfactual explanations showing "what-if" scenarios
- Visual explanations for every prediction

### 4. **Interactive Dashboard**
Built with Streamlit, featuring:
- Real-time risk segmentation (High/Medium/Low risk customers)
- Individual customer analysis with recommendations
- Model performance metrics and visualizations
- Business impact calculator with ROI projections

### 5. **Production-Ready Code**
- Proper train/test splits with stratification
- SMOTE for handling class imbalance
- Comprehensive data preprocessing pipeline
- Modular, well-documented code

## Project Architecture
```
┌─────────────────┐
│  Raw Data       │
│  (Telco Churn)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Data Preprocessing &       │
│  Feature Engineering        │
│  - Cleaning                 │
│  - RFM metrics              │
│  - Temporal features        │
│  - Encoding                 │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Model Training             │
│  - Random Forest            │
│  - XGBoost                  │
│  - LightGBM                 │
│  - MLflow tracking          │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Explainability             │
│  - SHAP analysis            │
│  - Counterfactuals          │
│  - Feature importance       │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Streamlit Dashboard        │
│  - Real-time predictions    │
│  - Customer analysis        │
│  - Business impact          │
└─────────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.12+
- pip

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
Dataset is included in `data/raw/` or download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## 💻 Usage

### 1. Run Data Preprocessing
```bash
jupyter notebook notebooks/02_data_preprocessing_and_feature_engineering.ipynb
```

### 2. Train Models
```bash
jupyter notebook notebooks/03_model_training_and_evaluation.ipynb
```

### 3. Generate Explainability Reports
```bash
jupyter notebook notebooks/04_explainability_and_counterfactuals.ipynb
```

### 4. Launch Interactive Dashboard
```bash
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`

### 5. View MLflow Experiments
```bash
mlflow ui
```

Open `http://localhost:5000` to view experiment tracking.

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | **0.79** | **0.55** | **0.71** | **0.62** | **0.835** |
| XGBoost | 0.78 | 0.54 | 0.69 | 0.61 | 0.828 |
| LightGBM | 0.78 | 0.53 | 0.70 | 0.60 | 0.830 |
| Logistic Regression | 0.76 | 0.51 | 0.65 | 0.57 | 0.810 |

**Key Insights:**
- Random Forest selected as production model (best ROC AUC)
- Churn rate: ~26% (imbalanced dataset)
- SMOTE applied for handling class imbalance
- Top predictive features: tenure, monthly charges, contract type

## 🛠️ Technologies Used

### Machine Learning
- **scikit-learn**: Model training, preprocessing, evaluation
- **XGBoost**: Gradient boosting
- **LightGBM**: Fast gradient boosting
- **imbalanced-learn**: SMOTE for class balancing

### Explainability
- **SHAP**: Feature importance and local explanations
- **DICE-ML**: Counterfactual explanations

### Data Processing
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### Visualization
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts in dashboard

### Dashboard & Deployment
- **Streamlit**: Interactive web application
- **MLflow**: Experiment tracking and model registry

### Development
- **Jupyter**: Notebooks for experimentation
- **Git**: Version control

## Project Structure
```
customer-churn-project/
│
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Processed data and splits
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing_and_feature_engineering.ipynb
│   ├── 03_model_training_and_evaluation.ipynb
│   └── 04_explainability_and_counterfactuals.ipynb
│
├── models/
│   ├── best_model.pkl               # Trained Random Forest model
│   ├── model_metadata.pkl           # Model metrics and info
│   └── *.png                        # Visualizations
│
├── dashboard/
│   └── app.py                       # Streamlit dashboard
│
├── mlruns/                          # MLflow experiment tracking
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```


## Future Enhancements

- [ ] Deploy to cloud (AWS/GCP/Azure)
- [ ] Add A/B testing framework for interventions
- [ ] Implement real-time streaming predictions with Kafka
- [ ] Add customer segmentation with clustering
- [ ] Build REST API with FastAPI
- [ ] Add automated retraining pipeline
- [ ] Implement drift detection

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- SHAP Library: For explainability framework
- Streamlit: For amazing dashboard capabilities

---
pandas==2.0.3
numpy<2
scikit-learn==1.3.0
xgboost==1.7.6
lightgbm==4.0.0
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0
streamlit==1.25.0
mlflow==2.5.0
dice-ml==0.10
shap==0.42.1
imbalanced-learn==0.11.0
fastapi==0.100.0
uvicorn==0.23.1
jupyter
