import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from decisiontree import DecisionTree
from logistic_regression import LogisticRegression
from adaboost import Adaboost
import argparse
from preprocess_data import preprocess_data
import seaborn as sns
import matplotlib.pyplot as plt

def accuracy_score(y_pred, y):
    return np.sum(y_pred == y) / len(y)

def recall_score(y_pred, y):
    true_positive = np.sum((y == 1) & (y_pred == 1))
    false_negative = np.sum((y == 1) & (y_pred == 0))
    
    if true_positive + false_negative == 0:
        return 0  # handle division by zero
    else:
        return true_positive / (true_positive + false_negative)

def precision_score(y_pred, y):
    true_positive = np.sum((y == 1) & (y_pred == 1))
    false_positive = np.sum((y == 0) & (y_pred == 1))
    
    if true_positive + false_positive == 0:
        return 0  # handle division by zero
    else:
        return true_positive / (true_positive + false_positive)

def f1_score(y_pred, y):
    precision = precision_score(y_pred, y)
    recall = recall_score(y_pred, y)

    if precision + recall == 0:
        return 0  # handle division by zero
    else:
        return 2 * precision * recall / (precision + recall)

def z_score_normalize(df_train):
    return (df_train - df_train.mean()) / df_train.std()

def calculate_metrics(y_pred, y):
    false_positive = np.sum((y == 0) & (y_pred == 1))
    false_negative = np.sum((y == 1) & (y_pred == 0))
    true_positive = np.sum((y == 1) & (y_pred == 1))
    true_negative = np.sum((y == 0) & (y_pred == 0))
    
    recall = true_positive / (true_positive + false_negative)
    precision = true_positive / (true_positive + false_positive)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)

    return precision, recall, f1_score, accuracy

def label_encode_column(df_train, column_name):
     unique_values = df_train[column_name].unique()
     mapping = {unique_values[i]: i for i in range(len(unique_values))}
     return df_train[column_name].replace(mapping)

def main():
    parser = argparse.ArgumentParser(description='Run different machine learning algorithms on the dataset.')
    parser.add_argument('--algorithm', choices=['decision_tree', 'logistic_regression', 'adaboost'], required=True, help='Choose the algorithm to run.')

    args = parser.parse_args()

    data = pd.read_csv('../data/updated_df_train_file.csv')

    if args.algorithm == 'logistic_regression':
        np.random.seed(0)
        data = preprocess_data(data)
        data = data.sample(frac=1).reset_index(drop=True)

    y = data['class']
    if args.algorithm == 'adaboost':
        y[y == 'e'] = -1
        y[y == 'p'] = 1
    else:
        y[y == 'e'] = 0
        y[y == 'p'] = 1
    
    y = y.astype(int)
    x = data.drop('class', axis=1)

    if args.algorithm != "logistic_regression":
        x = pd.get_dummies(x)

    split_index = int(np.ceil(2/3 * len(data)))

    # two different ways of splitting
    x_train, x_valid, y_train, y_valid = x.iloc[:split_index, :], x.iloc[split_index:, :], y.iloc[:split_index], y.iloc[split_index:]
    # x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)

    x_train = np.array(x_train)
    x_valid = np.array(x_valid)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)

    if args.algorithm == 'decision_tree':
        print('Decision Tree Results:\n')
        tree = DecisionTree()
        y_valid_pred = tree.myDT(x_train, y_train, x_valid)

    elif args.algorithm == 'logistic_regression':
        print('Logistic Regression Results:\n')
        learning_rate = 0.01
        epochs = 10000
        small_value = 2 ** (-32)
        logistic = LogisticRegression(learning_rate, epochs, small_value)
        logistic.run_logistic_regression(x_train, y_train, x_valid, y_valid)
        y_valid_pred = logistic.predict(x_valid)
        y_train_pred = logistic.predict(x_train)

    elif args.algorithm == 'adaboost':
        print('Adaboost Results:\n')
        clf = Adaboost(n_clf=50)
        clf.fit(x_train, y_train)
        y_train_pred = clf.predict(x_train)
        y_valid_pred = clf.predict(x_valid)

    else:
        print('Invalid algorithm choice. Please choose from: decision_tree, logistic_regression, adaboost')
        return

    # metrics
    acc = accuracy_score(y_valid_pred, y_valid)
    precision = precision_score(y_valid_pred, y_valid)
    recall = recall_score(y_valid_pred, y_valid)
    f1 = f1_score(y_valid_pred, y_valid)

    print("Validation Metrics:")
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1 Score: ', f1)
    print('Accuracy: ', acc)
    print(" ")

    if args.algorithm != 'decision_tree':
        acc = accuracy_score(y_train_pred, y_train)
        precision = precision_score(y_train_pred, y_train)
        recall = recall_score(y_train_pred, y_train)
        f1 = f1_score(y_train_pred, y_train)

        print("Training Metrics:")
        print('Precision: ', precision)
        print('Recall: ', recall)
        print('F1 Score: ', f1)
        print('Accuracy: ', acc)

    confusion = confusion_matrix(y_valid, y_valid_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

if __name__ == "__main__":
    main()


