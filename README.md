# 🏏 IPL Data Science: Professional Predictive Pipeline (2008-2020)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optimization-Optuna-blue.svg)](https://optuna.org/)

## 📋 Project Objective
This project implements a **senior-level analytical pipeline** for the Indian Premier League (IPL). We move beyond basic visualization to establish statistical rigor, engineering situational leverage features (CRR, RRR), and optimizing a win-probability predictor for second-inning chases.

---

## 📂 Advanced Notebook Architecture

1.  **[01_IPL_EDA.ipynb](notebooks/01_IPL_EDA.ipynb)**: Data Quality & Profiling (Missingno analysis, Skewness/Kurtosis, Outlier detection).
2.  **[02_Match_Analysis.ipynb](notebooks/02_Match_Analysis.ipynb)**: Macro-Trends (Chi-square toss tests, seasonal aggression trends, venue scoring bias).
3.  **[03_Ball_by_Ball_Analysis.ipynb](notebooks/03_Ball_by_Ball_Analysis.ipynb)**: Micro-Telemetry (U-shaped scoring curves, danger overs, pressure matrix visualization).
4.  **[04_Feature_Engineering.ipynb](notebooks/04_Feature_Engineering.ipynb)**: Domain Logic Engine (Mathematical derivation of RRR and situational pressure).
5.  **[05_Modeling.ipynb](notebooks/05_Modeling.ipynb)**: Predictive Intelligence (Stratified RF, Optuna tuning, win-probability simulation).

---

## 🛠️ Key Technical Features

### 🔍 Statistical Rigor
- **Chi-Square Tests**: Validated that toss winners are not statistically significant match-winning predictors.
- **Distribution Profiling**: Identified right-skewed scoring events using KDE and IQR outlier bounds.

### ⚙️ Feature Engineering
- **Pressure Index**: Derived a custom metric: $Pressure = \frac{RRR}{CRR + 1}$.
- **Resource Tracking**: Real-time monitoring of balls left and wickets in hand across 193k data points.

### 🤖 Machine Learning
- **Infrastructure**: Custom Sklearn `Pipelines` with `ColumnTransformer` for production-ready inference.
- **Optimization**: Bayesian Search with **Optuna**, achieving robust ROC-AUC through Stratified 5-Fold validation.

---

## 🚀 Environment Setup
```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Register Jupyter Kernel
python -m ipykernel install --user --name=ipl-venv --display-name "IPL Project (venv)"
```

---

## 📊 Business Insights
- **Aggression Trend**: IPL scoring velocity has increased by ~10% since 2008.
- **Venue Personalities**: Stadiums like Bangalore (Chinnaswamy) are identified as extreme high-scoring outliers requiring unique defensive strategies.
- **The Chase**: Required Run Rate (RRR) decay curves demonstrate that the tipping point for most teams occurs around Over 15, regardless of wickets in hand.
