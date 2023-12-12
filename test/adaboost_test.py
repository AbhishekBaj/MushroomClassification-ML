import numpy as np  
import pandas as pd
from sklearn.model_selection import train_test_split
from adaboost import Adaboost


def accuracy_score(y_pred, y):
    return np.sum(y_pred == y) / len(y)

def recall_score(y_pred, y):
    true_positive = np.sum((y == 1) & (y_pred == 1))
    false_negative = np.sum((y == 1) & (y_pred == 0))
    return true_positive / (true_positive + false_negative)

def precision_score(y_pred, y):
    true_positive = np.sum((y == 1) & (y_pred == 1))
    false_positive = np.sum((y == 0) & (y_pred == 1))
    return true_positive / (true_positive + false_positive)

def f1_score(y_pred, y):
    precision = precision_score(y_pred, y)
    recall = recall_score(y_pred, y)
    return 2 * precision * recall / (precision + recall)

def label_encode_column(df_train, column_name):
    unique_values = df_train[column_name].unique()
    mapping = {unique_values[i]: i for i in range(len(unique_values))}
    return df_train[column_name].replace(mapping)

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

data = pd.read_csv('updated_df_train_file.csv')

for column in data.columns:
    data[column] = label_encode_column(data, column)

X = data.iloc[:,1:23]
y = data.iloc[:,0]

y[y == 0] = -1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train = z_score_normalize(X_train)
X_test = z_score_normalize(X_test)
clf = Adaboost(n_clf=5)
clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)
acc = accuracy_score(y_pred, y_test.values)
precision   = precision_score(y_pred, y_test.values)
recall      = recall_score(y_pred, y_test.values)
f1_score    = f1_score(y_pred, y_test.values)
#precision, recall, f1_score, accuracy = calculate_metrics(y_pred, y_test.values)
print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', f1_score)
print('Accuracy: ', acc)

acc_train = accuracy_score(clf.predict(X_train.values), y_train.values)
print('Training Accuracy: ', acc_train)

