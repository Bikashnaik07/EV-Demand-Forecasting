# EV-Demand-Forecasting
Electric Vehicle Population Analysis and Prediction (Washington State, 2017–2024)

#Project Overview
This project analyzes and forecasts electric vehicle (EV) adoption trends in Washington State using historical vehicle registration data (2017–2024) from the Washington State Department of Licensing (via Kaggle). The goal is to explore EV penetration, identify key adoption patterns, and develop predictive models for future EV and charging demands.

##Week 1: Exploratory Data Analysis
We kick-started with understanding the dataset, identifying key metrics, and visualizing the ongoing EV revolution.

###Week 1 Deliverables:
Data loading, cleaning, and initial inspection
Monthly registration trends of BEVs and PHEVs
County-wise aggregation of vehicle counts
EV penetration rate over time
Time-series visualizations using Matplotlib and Seaborn

##Week 2: Forecasting Model Development
Building on our Week 1 insights, we now transition from observation to prediction using machine learning.

###Week 2 Deliverables:
Preprocessing & Outlier Handling:
Cleaned anomalies, handled missing values, and encoded categorical features.
Feature Engineering:
Extracted relevant features such as vehicle type, model year, county-wise distributions, and time variables to strengthen the model's predictive power.
Model Training:
Trained a Random Forest Regressor to predict future EV adoption using historical patterns. Hyperparameters were fine-tuned using RandomizedSearchCV.
Model Evaluation:
Evaluated predictions using MAE, RMSE, and R² metrics to quantify model performance.
Forecasting:
Generated short-term adoption forecasts for both single-county and multi-county levels.
Persistence:
Saved the trained model using joblib for deployment and reuse.

#Data Source:
Washington State Department of Licensing (via Kaggle)
Dataset: Electric Vehicle Population Size History by County
