# Customer Churn Prediction Using Neural Networks

This project predicts customer churn for a telecom company using a neural network. The model identifies at-risk customers, enabling the business to take proactive retention actions.

## Dataset

The dataset includes customer information such as account details, services subscribed, and demographic features. The target variable is `Churn`, indicating whether a customer has left the service.

## Data Preprocessing & EDA

* Conducted thorough **Exploratory Data Analysis (EDA)** with visualizations (histograms, box plots, correlation heatmaps) to identify key drivers of churn.
* Converted categorical 'Yes'/'No' columns into numerical values (1/0) and performed one-hot encoding on multi-category features.
* Split data into features (`X`) and target (`y`), followed by an 80%-20% train-test split with stratification to preserve class distribution.
* Applied **SMOTE (Synthetic Minority Over-sampling Technique)** to the training data to handle class imbalance by generating synthetic samples for the minority class.
* Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) to a 0-1 range using `MinMaxScaler`.

## Model Building

A neural network was implemented using Keras with the following architecture:

* Two hidden layers with ReLU activation and dropout for regularization.
* A Sigmoid-activated output layer for binary classification.
* The model was compiled with the Adam optimizer and binary cross-entropy loss.

## Hyperparameter Tuning

* Used **Keras Tuner** ('RandomSearch') to optimize the number of units in hidden layers, dropout rates, and the learning rate.
* Applied **EarlyStopping** during the final training of the best model to prevent overfitting and restore the best weights.

## Evaluation

The optimized model was evaluated on the unseen, imbalanced test set:

* **Test Accuracy:** **73.56%**
* **AUC (Area Under ROC Curve): 0.80**, indicating a strong ability to distinguish between classes.
* Non-churn customers (class 0) were predicted with high precision (0.87) and good recall (0.76).
* Churn customers (class 1) predictions showed a trade-off, with a **recall of 0.68** (identifying 68% of actual churners) and a **precision of 0.50**.
* Plotted **Loss/Accuracy curves, ROC curve, and Precision-Recall curve** to provide a comprehensive view of model performance.

## Conclusion

The project successfully demonstrates the use of a neural network with SMOTE to build a robust churn prediction model. A key finding was the difference between the high validation accuracy (achieved on the SMOTE-balanced data) and the more realistic test accuracy of 73.56% (on the original, imbalanced data), which highlights the model's true generalizability.

The model's **recall of 0.68 for churners** is valuable for business, as it allows for proactive engagement with a majority of at-risk customers. Future work could focus on optimizing the classification threshold to balance the trade-off between precision and recall.

## Libraries Used

* `pandas`, `numpy` for data manipulation
* `tensorflow`, `keras` for model building
* `keras_tuner` for hyperparameter tuning
* `matplotlib`, `seaborn` for visualization
* `sklearn` for metrics and data processing
* `imblearn` for handling class imbalance with SMOTE


The dataset used in this project is the Telecom Customer Churn dataset, which can be accessed via the following link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
