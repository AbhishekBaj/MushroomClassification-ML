# Machine Learning Final Project: Mushroom Classification

# Project Overview
This project implements and evaluates three machine learning algorithms: Decision Tree, Adaboost, and Logistic Regression. The primary goal is to compare the performance of these algorithms on the mushroom dataset. The code provides functionality for data preprocessing, model training, and evaluation metrics calculation.

# Prerequisites
Make sure you have the following dependencies installed:

numpy
pandas
scikit-learn
matplotlib
seaborn

# Code Structure

decisiontree.py: Contains the implementation of the Decision Tree algorithm.

adaboost.py: Implements the Adaboost algorithm.

logistic_regression.py: Implements Logistic Regression with gradient descent.

preprocess_data.py: Contains data preprocessing functions.

README.md: Project documentation.

data/updated_df_train_file.csv: Input dataset.

# Usage

Run the main script:

python main.py --algorithm <algorithm_name>
Replace <algorithm_name> with one of the following choices: 'decision_tree', 'logistic_regression', 'adaboost'.

Dataset
The dataset is expected to be in CSV format and located at data/updated_df_train_file.csv. 

Results
After running the script, the program will output metrics such as Precision, Recall, F1 Score, and Accuracy for both validation and training datasets. Additionally, a Confusion Matrix will be displayed for the validation dataset.

Additional Components
DecisionStump Class
The DecisionStump class implements a simple decision stump, which is a weak learner used in the Adaboost algorithm. It is a binary classifier that makes decisions based on a single feature and threshold.

Adaboost Class
The Adaboost class is an implementation of the Adaboost algorithm, which combines the predictions of multiple weak learners (decision stumps) to create a strong classifier.

LogisticRegression Class
The LogisticRegression class implements logistic regression using gradient descent. It is designed for binary classification tasks.
