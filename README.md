# flight_delay_prediction
**Introduction**

This repository implements a two-stage machine learning system to predict flight arrival delays using historical flight and weather data.

**Project Structure**

**data/:** Stores instructions to access and structure the original flight and weather data needed for the project.

**notebooks/:** Contains Jupyter Notebooks for data preprocessing, a CSV file with the merged datasets from preprocessing, exploratory data analysis (EDA), classification model training, and regression model training.

**results/:** Three CSV files that stores performance measures for the siloed classification models, regression models as well the performance results of the 4 pipelines built between the best performing classification model, the Decision Tree Classifier, and each regressor that predicted on the delayed output of the classifier.

There are also two python scripts that can be inserted into their respective classifier or regressor's code to get the performance measures of that model (if you want to adjust the model and see its new performance, reinsert the script into model's JupyterNotebook).

**README.md:** This file! 

**Data Preprocessing**

The datapreproc.ipynb notebook performs the following tasks:

    Loads flight data for years 2016 and 2017 for specified airports (ATL, CLT, DEN, DFW, EWR, IAH, JFK, LAS, LAX, MCO, MIA, ORD, PHX, SEA, SFO).
    
    Loads weather data for the corresponding airports and timeframes.
 
    Selects relevant features for analysis (windspeed, wind direction, precipitation, visibility, pressure, cloud cover, dew point, wind gust, temperature, wind chill, humidity, departure delay, and arrival delay) from the intensive list of features provided in the weather data.
    
    Handles missing values and filters the data set of flights that we have the corresponding origin airport weather information for. These are the features that will be trained on later. 

    Merges flight and weather data based on airport codes and timestamps, and date. This is done by mapping each flight to the closest hourly weather data.

**Exploratory Data Analysis (EDA)**

    The eda.ipynb notebook explores the merged dataset to understand:
    
    Distribution of arrival delay times.
    
    Correlations between arrival delays and other relevant features to see if there is any one feature that has a strong, direct linear relationship with the delay of a flight. If any one feature is found to have a strong correlation with delay, then this is not a problem for machine learning, which works in finding relationships within the data in higher dimensions.
    
    Bivariate visualizations of ArrDelayMinutes and each of the features to observe anu potential patterns or relationships.

**Data Resampling**
   **Train-Test Split:** The merged dataset was split into training and testing sets using an 80/20 ratio. This ensures a representative portion of the data is used for model training and evaluation.

**Target Variable Selection:**
    Classification: The "ArrDEl15" column, indicating arrival delay exceeding 15 minutes (binary classification), was chosen as the target variable (y) for the classification model.
    
Regression: The "ArrDelayMinutes" column, representing the actual delay duration in minutes, was used as the target variable (y) for the regression model. However, it was also included as a feature (predictor) in the training data (X) initially so that the dataset can be rebalanced and then it could be extrapolated out when creating the final rebalanced training set for regression models.
    
**SMOTE for Imbalance Correction (Training Set Only):**
To address the potential bias caused by imbalanced data in the training set, the Synthetic Minority Oversampling Technique (SMOTE) was applied. SMOTE generates synthetic samples for the minority class, flights with significant delays, to create a more balanced distribution within the training data, X_train and y_train.

SMOTE is not applied to the testing set, X_test and y_test, to prevent data leakage. Data leakage occurs when information from the testing set influences the training process, leading to an overly optimistic performance evaluation.

**Creating Separate Training Sets for Classification and Regression:**

    For the classification model:
    The original "ArrDelayMinutes" column was dropped from X_train as it would have made the model overfit on that one feature, which indicates the delay of the flight.
    The prepared X_train and the original y_train containing "ArrDEl15" were merged to create the final classification training set.
    
    For the regression model:
    The "ArrDelayMinutes" column that was retained in X_train is stored in its own y_train_reg data frame as it's is needed as the target variable for regression feature for predicting the delay duration.
    The prepared X_train and the newly created y_train_reg containing only "ArrDelayMinutes" were merged to create the final regression training set.

    Testing Sets: Similar procedures were followed to create the testing sets (X_test and y_test) for both classification and regression models, ensuring consistency with the training sets.

    Respective X_train and y_train as well as X_test and y_test data frams were concatenated together into a CSV file to easily read from and use for the training of each model. These can be accessed in the links provided in the data directory to the original flight and weather data and merged dataset that is used later to train the regressor in the 2-stage predictive pipeline on only delayed flight information.

**Classification Models**

The following classification models were trained and evaluated in separate notebooks (logistic_model.ipynb, decisiontree.ipynb, extratrees.ipynb, xgboost.ipynb, and randomforest.ipynb):

    Logistic Regression: A model for binary classification problems like flight delay prediction that
estimates the probability of a flight being delayed based on features (weather, distance) using a sigmoid function.
    
    Decision Tree: A tree-based classifier that learns a series of decision rules by iteratively splitting the data based on features. At each split, the algorithm chooses the feature and threshold that maximizes a splitting criterion, such as Gini Index or information gain, to create the most homogeneous child nodes in terms of the target variable, delayed vs. on-time flights. This process continues until a stopping criterion is met, resulting in a tree-like structure that can classify new data points.
    
    Extra Trees Classifier: An ensemble model of decision trees that improves upon a single decision tree.
    
    XGBoost: A gradient boosting algorithm that builds an ensemble of decision trees sequentially, where each tree focuses on correctly classifying flights that prior trees struggled with.
    
    Random Forest: A ensemble method that averages predictions from multiple decision trees for improved robustness. It introduces randomness by selecting features at each split point within the trees, potentially reducing overfitting and improving the model's ability to generalize to unseen data.

**Evaluation Metrics**

Each classification model is evaluated using...

**1. Accuracy:**

Formula: Accuracy = (True Positives + True Negatives) / (Total Samples)

True Positives (TP): Number of flights correctly classified as delayed.

True Negatives (TN): Number of flights correctly classified as on-time.

Total Samples: Total number of flights in the dataset.

Interpretation: Accuracy reflects the overall proportion of correctly classified flights, regardless of class. However, in imbalanced datasets like flight delays where on-time flights might be much more frequent, a high accuracy might not necessarily indicate the model's ability to accurately identify actual delays.

**2. Precision:**

Formula: Precision = TP / (TP + False Positives)

False Positives (FP): Number of flights incorrectly classified as delayed.

Interpretation: Precision measures the proportion of flights predicted as delayed that were actually delayed. It's the best performance indicator when the cost of misclassifying a flight as delayed is high, for example an unnecessary passenger rebooking or missed connections due to reassigned gates. A high precision ensures a low false alarm rate.

**3. Recall:**

Formula: Recall = TP / (TP + False Negatives)

False Negatives (FN): Number of flights incorrectly classified as on-time (actual delayed flights).

Interpretation: Recall measures the proportion of actual delayed flights that were correctly predicted as delayed. It's important when missing a delayed flight prediction can have significant consequences, such as missed connections for passengers. A high recall ensures a low miss rate for actual delays.

**4. F1-Score:**

Formula: F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

Interpretation: F1-score is the mean of precision and recall, combining their strengths into a single metric. It provides a balance between the model's ability to identify actual delays, recall, and avoid false alarms, precision. F1-score is often preferred for imbalanced datasets as it doesn't favor the majority class and ensures the model performs well on both identifying actual delays and avoiding false alarms.

I evaluated my models on the highest F1-score, which was the Decision Tree Classifier. This was used as the starting stage of my 2-stage pipeline later.

You can adjust hyperparameters within each model to potentially improve model performance. 

**Regression Models**

The following regression models were trained and evaluated in separate notebooks (linearregressor.ipynb, extratreesregressor.ipynb, xgboostregressor.ipynb, and randomforestregressor.ipynb):

    Linear Regression: A model that learns a linear relationship between features and arrival delay times. This model tries to fit a straight line through your data points, aiming to minimize the distance between the line and the actual delay times. It assumes a linear relationship between features (weather, humidity, pressure, etc.) as an x-matrix input  and arrival delays (y).
    
    Extra Trees Regressor: Ensemble regression model based on a forest of individual decision trees, each making predictions based on splits on different features and thresholds of the data. Extra Trees combines these predictions from multiple trees to create a more robust and accurate overall prediction for arrival delays.
    
    XGBoost Regressor: Gradient boosting algorithm for regression tasks. It builds an ensemble of decision trees sequentially, where each tree focuses on improving the predictions of the previous one. This is achieved by learning from the "gradients" or errors of the prior trees.
    
    Random Forest Regressor: Ensemble regression model using random forests. Similar to Extra Trees, it creates a forest of decision trees for predicting arrival delays. However, when making predictions, it randomly selects features at each split in the trees, potentially reducing overfitting.

**Evaluation Metrics**

Each regression model is evaluated using:

    Mean Squared Error (MSE): Measures the average squared difference between the predicted delay times and the actual delay times in the testing set. Lower MSE values indicate a better fit between the model's predictions and the real delays. While MSE reflects the average prediction error, it can be difficult to interpret directly in minutes due to squaring the errors. Higher MSE values suggest larger discrepancies between predicted and actual delays.
    
    Root Mean Squared Error (RMSE): The square root of the Mean Squared Error (RMSE = sqrt(MSE)). RMSE overcomes the interpretability issue of MSE by bringing the error units back to the original scale of delay minutes. A lower RMSE indicates better model performance, signifying that the model's predictions are, on average, closer to the actual delay times. 

    Mean Absolute Error (MAE): Measures the average of the absolute differences between the predicted delay times and the actual delay times in the testing set. Unlike MSE, it doesn't square the errors. MAE provides a more intuitive understanding of prediction error, directly reflecting the average difference between predicted and actual delays. This can be easier to interpret for business users.
    
    R-squared: Represents the proportion of variance, or spread, in the actual arrival delay times that can be explained by the model. It ranges from 0 to 1, with higher values indicating a better fit. R-squared provides insight into the model's ability to explain the relationship between features and arrival delays. While it's not a direct measure of prediction accuracy, a high R-squared can indicate that the model has learned meaningful patterns from the data. This can be valuable for understanding the factors that contribute to flight delays.

Focus on overall passenger experience and minimizing disruption: Choose the model with a lower RMSE.

Prioritize avoiding large delays and missed connections: Choose the model with a lower MAE.

**Feature importance** 

Feature Importance measures the relative influence of each feature on a model's prediction. Understanding which features contribute most to the model's performance allows us to identify the key factors driving the model's predictions and potentially train models faster and with less memory by focusing on the most impactful feature.


This project explores training the best performing pipeline model (Decision Trees + Extra Trees) with the top 6 features for each model within a pipeline. This approach offers several advantages such as ...

    Increased Efficiency: Training models with fewer features can significantly reduce training time and resource consumption.
    
    Potential Performance Improvement: By removing less informative features, we can sometimes improve model performance by reducing overfitting and focusing on the most relevant information.
    
This project demonstrates the trade-off between efficiency and performance. While selecting the top features often improves efficiency, it's crucial to ensure minimal degradation in model performance.  The pipeline_results.csv file provides insights into this trade-off for the specific models used in this analysis. The performance from training from the top 6 features barely affects the performance of the model from when it was trained on all 12 features.

The feature_importance.ipynb notebook outlines the code used for feature analysis. 







