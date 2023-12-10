import numpy as np

def calculate_metrics(y_pred, y):
    false_positive = np.sum((y == 0) & (y_pred == 1))
    false_negative = np.sum((y == 1) & (y_pred == 0))
    true_positive = np.sum((y == 1) & (y_pred == 1))
    true_negative = np.sum((y == 0) & (y_pred == 0))
    
    denominator = true_positive + false_negative
    recall = true_positive / denominator if denominator != 0 else 0
    
    denominator = true_positive + false_positive
    precision = true_positive / (true_positive + false_positive) if denominator != 0 else 0

    denominator = (precision + recall)
    f1_score = 2 * (precision * recall) / (precision + recall) if denominator != 0 else 0
    accuracy = (true_positive + true_negative) / len(y)

    return precision, recall, f1_score, accuracy

def confusion_matrix(y_true, y_pred, classes):
    num_classes = len(classes)
    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        confusion_mat[true_label, pred_label] += 1 
    return confusion_mat

