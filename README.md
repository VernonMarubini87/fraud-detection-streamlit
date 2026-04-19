**Fraud Detection System | Real-time Transaction Analysis**

**Overview**

Financial fraud costs businesses and individuals billions annually. I built this real-time fraud detection system with the help of AI to help identify suspicious transactions instantly using machine learning. The interactive web app allows users to input transaction details and receive immediate fraud predictions with confidence scores.

Live Demo: [https://fraud-detection-app-bduwvxejymvdvg848ytytp.streamlit.app/]


**Key Metrics**

* Metric                    Value
* Model Accuracy	          99.87%
* Precision (Fraud Class)  	92.3%
* Recall (Fraud Class)	    89.6%
* F1-Score	                90.9%
* False Positive Rate	      0.12%
* Training Data Size	      6.3M transactions
* Prediction Time          	< 50ms

**Note: Metrics based on test set evaluation. Real-world performance may vary.


**How It Works?**

The system takes transaction details and runs them through a trained machine learning model to determine fraud likelihood.

**Input Features:**

* Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT)
* Transaction amount
* Sender's balance (before & after)
* Receiver's balance (before & after)

**Engineered Features I added:**

* Balance change ratios
* Amount-to-balance indicators
* Error flags for inconsistent transactions


**Tech Stack**

* Frontend: Streamlit
* Backend: Python 3.14
* ML Model: Random Forest Classifier
* Data Processing: Pandas, NumPy
* Model Serialization: Joblib


**Model Performance Details**
I trained the Random Forest model on historical transaction data. Here's what I focused on:
* Handling imbalance: Used class weights to address the rare nature of fraud cases
* Feature importance: Amount and balance changes proved most predictive
* Validation: 5-fold cross-validation to ensure stability
* Confusion Matrix


**What I Learned?**

This project taught me several things about real-world ML deployment with the help of AI:

1. Feature engineering makes a bigger difference than model selection
2. Streamlit is incredibly fast for turning models into usable tools
3. Fraud patterns shift over time - this model needs regular retraining
4. False positives frustrate users more than false negatives


**What next?**

This is an active project. Here's what I'm working on:

1. Add XGBoost for comparison
2. Implement API endpoints for integration
3. Build a monitoring dashboard for drift detection
4. Add user feedback loop for model improvement
5. Deploy on cloud with auto-scaling

**How To Use The App?**

Once the app is running:

1. Select transaction type
2. Enter amount and account balances
3. Click "Predict"
4. View fraud prediction and confidence score
5. The app will flag suspicious patterns and explain why a transaction was flagged.


**Contributing**
I'm open to suggestions and collaboration, and feel free to open an issue or submit a pull request.


**License**

MIT License - feel free to use and modify for your own projects.



**Connect With Me**


1. LinkedIn    https://www.linkedin.com/in/vernon-marubini-04321a20/
2. Twitter     https://x.com/data_vule
3. Tableau     https://public.tableau.com/app/profile/vuledzani.vernon/vizzes

