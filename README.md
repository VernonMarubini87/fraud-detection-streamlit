# Fraud Detection System | Real-Time Transaction Risk Scoring

An end-to-end machine learning system that flags suspicious financial transactions in real time. Built on 6.3M+ historical transactions, the model scores incoming transactions for fraud risk and surfaces that score through an interactive web app.

**Live demo:** https://fraud-detection-app-bduwvxejymvdvg848ytytp.streamlit.app/

---

## Overview

Financial fraud costs businesses and individuals billions annually, and most of that damage happens because detection lags the transaction — by the time a human reviews it, the money is gone. This project tackles that gap with a Random Forest classifier trained on transaction-level data, wrapped in a Streamlit app so a transaction can be scored and explained in under a second.

The interactive app takes in transaction details and returns:
- A fraud / not-fraud prediction
- A confidence score
- The transaction attributes that drove the flag

## Key Metrics

| Metric | Value |
|---|---|
| Model Accuracy | 99.87% |
| Precision (Fraud Class) | 92.3% |
| Recall (Fraud Class) | 89.6% |
| F1-Score | 90.9% |
| False Positive Rate | 0.12% |
| Training Data Size | 6.3M transactions |
| Prediction Time | < 50ms |

*Metrics based on test-set evaluation. Real-world performance on live data may vary.*

## How It Works

The system takes transaction details and runs them through a trained Random Forest model to determine fraud likelihood.

**Input features:**
- Transaction type (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEPOSIT`)
- Transaction amount
- Sender's balance (before & after)
- Receiver's balance (before & after)

**Engineered features:**
- Balance change ratios (`BalanceDiffOrig`, `BalanceDiffDest`) — these were identified in EDA as strong fraud signals and are included in the final trained model
- Amount-to-balance indicators
- Error flags for inconsistent transactions (e.g. balances that don't reconcile with the transaction amount)

## Project Structure

```
fraud-detection-streamlit/
│
├── app.py                     # Streamlit application
├── analysis_model.ipynb       # EDA, feature engineering, model training & evaluation
├── fraud_detection_model.pkl  # Serialized trained model (joblib)
├── requirements.txt           # Python dependencies
├── .devcontainer/             # Dev container config for reproducible environment
└── README.md
```

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python 3.12
- **ML Model:** Random Forest Classifier (scikit-learn)
- **Data Processing:** Pandas, NumPy
- **Model Serialization:** Joblib

## Model Performance Details

The Random Forest model was trained on historical transaction data with a focus on:

- **Handling class imbalance:** Used class weights to address the rare nature of fraud cases rather than naively resampling, to keep the model calibrated on real-world fraud rates
- **Feature importance:** Transaction amount and balance-change features proved most predictive
- **Validation:** 5-fold cross-validation to confirm the model's stability across folds
- **Confusion matrix:** Reviewed to understand the precision/recall trade-off at the chosen decision threshold

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/VernonMarubini87/fraud-detection-streamlit.git
cd fraud-detection-streamlit

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## Using the App

1. Select transaction type
2. Enter amount and account balances (before & after, sender & receiver)
3. Click **Predict**
4. Review the fraud prediction, confidence score, and the flagged risk indicators

## What I Learned

Building this taught me several things about real-world ML deployment, working with AI tools throughout the process:

1. Feature engineering makes a bigger difference than model selection — the balance-change features moved the needle more than any hyperparameter tuning
2. Streamlit is an extremely fast path from a trained model to a usable, demoable tool
3. Fraud patterns shift over time, so a static model needs a retraining cadence, not a one-time deployment
4. False positives frustrate users more than false negatives — that trade-off shaped the decision threshold

## Roadmap

This is an active project. Next steps:

- [ ] Add XGBoost as a comparison model
- [ ] Expose the model via an API endpoint for system integration
- [ ] Build a monitoring dashboard for data/model drift detection
- [ ] Add a user feedback loop to capture mislabeled predictions
- [ ] Deploy on cloud infrastructure with auto-scaling

## Contributing

Open to suggestions and collaboration — feel free to open an issue or submit a pull request.

## License

MIT License — free to use and modify for your own projects.

## Connect With Me

- **LinkedIn:** [vernon-marubini-04321a20](https://www.linkedin.com/in/vernon-marubini-04321a20/)
- **X (Twitter):** [@data_vule](https://x.com/data_vule)
- **Tableau Public:** [vuledzani.vernon](https://public.tableau.com/app/profile/vuledzani.vernon/vizzes)


