# flight_delay_prediction
**Introduction**

This repository implements a two-stage machine learning system to predict flight arrival delays using historical flight and weather data.

**Project Structure**

**data/:** Stores instructions to access and structure the original flight and weather data needed for the project.

**notebooks/:** Contains Jupyter Notebooks for data preprocessing, a CSV file with the merged datasets from preprocessing, exploratory data analysis (EDA), classification model training, and regression model training.

**results/:** Two CSV files that stores performance measures for the classification models and regression models respectively as well as the python script to get the measures from each model (if you want to adjust the model and see its new performance, reinsert the script into model's JupyterNotebook)

**README.md:** This file! 

**Data Preprocessing**

The datapreproc.ipynb notebook performs the following tasks:

    Loads flight data for years 2016 and 2017 for specified airports (ATL, CLT, DEN, DFW, EWR, IAH, JFK, LAS, LAX, MCO, MIA, ORD, PHX, SEA, SFO).
    
    Loads weather data for the corresponding airports and timeframes.
    
    Merges flight and weather data based on airport codes and timestamps.
    
    Selects relevant features for analysis (windspeed, wind direction, precipitation, visibility, pressure, cloud cover, dew point, wind gust,     
    temperature, wind chill, humidity, departure delay, and arrival delay).
    
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

**Conclusion**

In this project, I  developed a two-stage predictive model to classify and predict flight delays using historical flight and weather data. The project involved data cleaning and preprocessing to merge flight data with corresponding hourly weather conditions at the departure airport. I explored various classification and regression models, ultimately selecting a Decision Tree classifier and an XGBoost regressor based on their performance metrics.

The Decision Tree classifier, with a maximum depth of 8 and utilizing the Gini criterion, achieved an accuracy of 81.23%. The XGBoost regressor demonstrated a strong predictive capability with an R-squared value of 0.94. These models were integrated into a two-stage pipeline where the classifier first identified delayed flights, and the regressor then predicted the delay duration for these flights.

For evaluation, I split the data into 80% for training and 20% for testing. The testing set that the regressor predicted on consisted of 95,982 rows of delayed flights classified in the previous stage of the model, which is 25.92% of the classifier's testing set and 5.18% of the original dataset. This confirmed the model's effectiveness, with the regressor achieving an improved R-squared value of 0.94 and an RMSE of 16.77 minutes. The high R-squared value of 0.94 indicates that the model explains 94% of the variance in flight delays, showing strong predictive power. The RMSE of 16.77 minutes means that, on average, the model's delay predictions are within 16.77 minutes of the actual delays. This level of accuracy can help airlines better manage schedules and resources, leading to improved operational efficiency and enhanced customer satisfaction.



