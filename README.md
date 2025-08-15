# Customer Churn Prediction Using Neural Networks

This project aims to predict customer churn for a telecom dataset using neural networks, allowing businesses to identify at-risk customers and take proactive retention actions.

## Dataset

The dataset includes customer information such as account details, services subscribed, and demographic features. The target variable is `Churn`, indicating whether a customer has left the service.

## Data Preprocessing

* Converted categorical 'Yes'/'No' columns into numerical values (1/0) for model compatibility.
* Split data into features (`X`) and target (`y`), followed by an 80%-20% train-test split with stratification to preserve class distribution.
* Calculated class weights to address imbalance between churn and non-churn classes.

## Model Building

A neural network was implemented using Keras:

* Two hidden layers with ReLU activation and dropout for regularization.
* Sigmoid-activated output layer for binary classification.
* Compiled with Adam optimizer and binary cross-entropy loss.
* Trained with class weights applied to improve handling of minority class.

## Hyperparameter Tuning

* Used **Keras Tuner** ('RandomSearch') to optimize the number of units, dropout rates, and learning rate.
* Applied EarlyStopping to prevent overfitting.
* Best model trained and validated on the training set.

## Evaluation

* **Test Accuracy:** 78.54%
* Non-churn customers (class 0) predicted with higher precision (0.83) and recall (0.89).
* Churn customers (class 1) predictions were more challenging (precision 0.62, recall 0.51).
* Weighted F1-score: 0.78, indicating balanced performance overall.
* Loss and accuracy curves over epochs were plotted to monitor training and validation performance.

## Conclusion

The project demonstrates effective use of neural networks and hyperparameter tuning to predict customer churn. While predictions for non-churn customers are strong, identifying churners remains more challenging due to class imbalance. Further improvements could involve advanced architectures or additional feature engineering.

## Libraries Used

* `pandas`, `numpy` for data manipulation
* `tensorflow`, `keras` for model building
* `keras_tuner` for hyperparameter tuning
* `matplotlib` for visualization
* `sklearn` for metrics and train-test split

