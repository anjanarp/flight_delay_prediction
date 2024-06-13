# flight_delay_prediction
Introduction

This repository implements a two-stage machine learning system to predict flight arrival delays using historical flight and weather data.

**Project Structure**

data/: Stores instructions to access and structure the original flight and weather data needed for the project.
notebooks/: Contains Jupyter Notebooks for data preprocessing, a CSV file with the merged datasets from preprocessing, exploratory data analysis (EDA), classification model training, and regression model training.
results/ Two CSV files that stores performance measures for the classification models and regression models respectively as well as the python script to get the measures from each model (if you want to adjust the model and see its new performance, reinsert the script into model's JupyterNotebook)
README.md: This file! 

**Data Preprocessing**

The datapreproc.ipynb notebook performs the following tasks:

Loads flight data for years 2016 and 2017 for specified airports (ATL, CLT, DEN, DFW, EWR, IAH, JFK, LAS, LAX, MCO, MIA, ORD, PHX, SEA, SFO).
Loads weather data for the corresponding airports and timeframes.
Merges flight and weather data based on airport codes and timestamps.
Selects relevant features for analysis (windspeed, wind direction, precipitation, visibility, pressure, cloud cover, dew point, wind gust, temperature, wind chill, humidity, departure delay, and arrival delay).
Handles missing values and potential data cleaning procedures (replace as needed).

**Exploratory Data Analysis (EDA)**

The eda.ipynb notebook explores the merged dataset to understand:

Distribution of arrival delay times.

Correlations between arrival delays and other relevant features.

Visualizations to identify potential patterns or relationships.

**Classification Models**

The following classification models were trained and evaluated in separate notebooks (logistic_model.ipynb, decisiontree.ipynb, extratrees.ipynb, xgboost.ipynb, and randomforest.ipynb):

Logistic Regression: A model for binary classification problems like flight delay prediction.

Decision Tree: A tree-based model that learns a series of decision rules to classify flights.

Extra Trees Classifier: An ensemble model of decision trees that improves upon a single decision tree.

XGBoost: A gradient boosting algorithm ombine predictions from multiple weaker models to create a stronger overall model.

Random Forest: A ensemble method that averages predictions from multiple decision trees for improved robustness.

**Evaluation Metrics**

Each classification model is evaluated using...

Accuracy: Proportion of correctly classified flights (delayed vs. not delayed).

Confusion Matrix: Visualizes the distribution of true vs. predicted delays.

Classification Report: Provides a detailed breakdown of model performance metrics (precision, recall, F1-score, support).

You can adjust hyperparameters within each model to potentially improve model performance. 

**Regression Models**

The following regression models were trained and evaluated in separate notebooks (linearregressor.ipynb, extratreesregressor.ipynb, xgboostregressor.ipynb, and randomforestregressor.ipynb):

Linear Regression: A model that learns a linear relationship between features and arrival delay times.

Extra Trees Regressor: Ensemble regression model based on decision trees.

XGBoost Regressor: Gradient boosting algorithm for regression tasks.

Random Forest Regressor: Ensemble regression model using random forests.

**Evaluation Metrics**

Each regression model is evaluated using:

Mean Squared Error (MSE): Average squared difference between predicted and actual delay times (lower is better).

Root Mean Squared Error (RMSE): Square root of MSE (easier to interpret in units of delay minutes).

R-squared: Proportion of variance in arrival delay explained by the model (higher is better).


**Results Summary**

THe Decision Tree using a Gini Index criterion with a max-depth of 8 had the highest accuracy of 81.2346% and performed the best among overall among all the classification models

The XGBoost Regressor performed the most accurately among all the models with a R-squared 0.935879 value.

**2-stage Predictive Pipeline**

The 2_state_predictor.ipynb script employs a decision tree classifier which predicts whether flights in the test set (y_test) will be delayed. These predictions are then used to filter the features in X_test. The filtered data is fed into an XGBoost Regressor. This regressor is trained only on delayed flight data from the original dataset. Its purpose is to predict the delay duration for flights identified as 'Delayed' by the decision tree classifier.

