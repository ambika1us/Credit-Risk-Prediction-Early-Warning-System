# Credit Risk Prediction – Early Warning System
A machine learning–powered Streamlit application that predicts the probability of loan default and classifies accounts into Low Risk, Watchlist, or High Risk categories. Designed as an early warning system for financial institutions, this tool helps monitor portfolio health and identify risky accounts before defaults occur.
# Key Features
- Single Record Prediction: Enter loan details manually to assess risk for an individual borrower.
- Batch Upload Mode: Upload a CSV file with multiple loan records for bulk risk assessment.
- Automated Preprocessing: Cleans input data, applies one-hot encoding, and aligns with the training schema.
# Risk Classification: Categorizes accounts into:
- Low Risk
- Watchlist (Early Warning)
- High Risk (Likely Default)
# Portfolio Dashboard:
- KPI metrics (High Risk %, Watchlist %, Average Default Probability)
- Risk category distribution (bar chart)
- Default probability distribution (histogram with thresholds)
- Threshold sensitivity analysis
# Tech Stack
- Python
- Streamlit (interactive UI)
- Scikit-learn / Random Forest (ML model)
- Joblib (model persistence)
- Pandas & NumPy (data handling)
- Matplotlib (visualizations)
