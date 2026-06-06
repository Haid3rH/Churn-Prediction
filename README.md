# 📉 ChurnIntel — Customer Churn Prediction

> *Not every customer who leaves sends a goodbye. This model catches them before they do.*

Built by **Haider Haroon** · [Live App →](https://churn-intel.streamlit.app/)

---

## What is this?

A machine learning pipeline that predicts whether a telecom customer is about to churn — based on their contract type, billing, services, and usage patterns. Trained on the Telco Customer Churn dataset, deployed as an interactive Streamlit app.

Put in a customer's details, get back a churn prediction with probability.

---

## How it works

```
Raw Customer Data (19 features)
    ↓
Data Cleaning  ←  fix TotalCharges whitespace, drop CustomerID
    ↓
Label Encoding  ←  all categorical columns encoded + saved as encoders.pkl
    ↓
SMOTE  ←  fix class imbalance (churners are a minority)
    ↓
3 Models Compared:
  • Decision Tree
  • Random Forest  ← winner (best CV accuracy)
  • XGBoost
    ↓
Random Forest → Saved as customer_churn_model.pkl
    ↓
Streamlit App → Live Predictions
```

---

## Results

| Model | CV Accuracy |
|-------|-------------|
| Decision Tree | Baseline |
| XGBoost | Strong |
| **Random Forest** | **Best ✅** |

Evaluated with accuracy score, confusion matrix, and full classification report on a held-out 20% test set.

---

## Key decisions worth noting

- **SMOTE** applied *after* the train/test split — avoids data leakage
- **Label encoders saved with pickle** so the app uses identical encoding at inference time
- **CustomerID dropped** — it's a unique key, zero predictive value
- **TotalCharges whitespace** replaced with `0.0` (11 new customers with no charges yet)

---

## Run it locally

```bash
# 1. clone the repo
git clone https://github.com/Haid3rH/ChurnIntel
cd ChurnIntel

# 2. install dependencies
pip install streamlit scikit-learn xgboost imbalanced-learn pandas numpy matplotlib seaborn

# 3. launch the app
streamlit run app.py
```

Or just use the live version → **[churn-intel.streamlit.app](https://churn-intel.streamlit.app/)**

---

## Stack

`Python 3.10` · `Scikit-learn` · `XGBoost` · `imbalanced-learn` · `Pandas` · `Streamlit` · `Seaborn` · `Pickle`

---

*Haider Haroon · [LinkedIn](https://linkedin.com/in/haider-haroon-8a0209306/) · [GitHub](https://github.com/Haid3rH)*
