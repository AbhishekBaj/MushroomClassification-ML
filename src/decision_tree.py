from statistics import mode
import pandas as pd
import numpy as np
from tree import Node
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
from metrics import calculate_metrics, confusion_matrix

def ID3():
    np.random.seed(0)
    df = preprocess_data()
    data = df.sample(frac=1).reset_index(drop=True)

    x_train, x_valid, y_train, y_valid = prepare_data(data)
    # x_train, x_valid = convert_to_onehot(x_train, x_valid)
    x_train, x_valid = convert_to_binary(x_train, x_valid)

    y_valid_pred = myDT(x_train, y_train, x_valid)
    precision, recall, f1_score, accur = calculate_metrics(y_valid, y_valid_pred)

    print("Validation Metrics:")
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1_score: ", f1_score)
    print("accuracy: ", accur)
    print(" ")

    confusion_mat = confusion_matrix(y_valid, y_valid_pred, np.unique(y_valid))
    print("Confusion Matrix:\n", confusion_mat)

    ## entropy vs feature plot
    # num_features = x_valid.shape[1]
    # entropies = []

    # for feature_index in range(num_features):
    #     entropy = calculate_entropy(y_train[x_train[:, feature_index] == 1])
    #     entropies.append(entropy)

    # plt.figure(figsize=(10, 6))
    # plt.bar(range(num_features), entropies, color='skyblue')
    # plt.xlabel('Feature Index')
    # plt.ylabel('Entropy')
    # plt.title('Entropy for Each Feature')
    # plt.show()

def myDT(x_train, y_train, x_valid):
    root = DTL(x_train, y_train, set(range(x_train.shape[1])), mode(y_train))
    y_valid_pred = traverse_tree(x_valid, root)
    return y_valid_pred

def DTL(X, Y, features, default):
    if len(X) == 0:
        return Node(value=default)
    elif np.unique(Y).size == 1:
        return Node(value=Y[0])
    elif len(features) == 0:
        return Node(value=default, probabilities=probabilities(Y))
    else:
        best = choose_attribute(X, Y, list(features) )
        tree = Node(feature=best)

        X_true, Y_true = X[X[:, best] == 1], Y[X[:, best] == 1]
        left_child = DTL(X_true, Y_true, features - {best}, mode(Y))
        tree.left = left_child

        X_false, Y_false = X[X[:, best] == 0], Y[X[:, best] == 0]
        right_child = DTL(X_false, Y_false, features - {best}, mode(Y))
        tree.right = right_child
        return tree

def choose_attribute(X, Y, feature_list):
    best_attribute = None
    best_entropy = float('inf')

    for feature in feature_list:
        entropy = calculate_entropy(Y[X[:, feature] == 1])
        if entropy < best_entropy:
            best_entropy = entropy
            best_attribute = feature
    return best_attribute

def convert_to_onehot(x_train, x_valid):
    df_train = pd.DataFrame(x_train)
    df_valid = pd.DataFrame(x_valid)

    df_train_encoded = pd.get_dummies(df_train, columns=df_train.columns)
    df_valid_encoded = pd.get_dummies(df_valid, columns=df_valid.columns)

    x_train_encoded = df_train_encoded.values
    x_valid_encoded = df_valid_encoded.values

    return x_train_encoded, x_valid_encoded

def convert_to_binary(x_train, x_valid):
    median = np.median(x_train, axis=0)
    x_train = x_train > median
    x_valid = x_valid > median
    return x_train, x_valid


def traverse_tree(X, tree):
    predictions = []
    for instance in X:
        predictions.append(predict_instance(instance, tree))
    predictions = np.array(predictions, dtype=object)
    return predictions

def predict_instance(instance, tree):
    if tree is None:
        return 0
    if tree.value:
        return tree.value

    if tree.feature and instance[tree.feature]:
        return predict_instance(instance, tree.left)
    else:
        return predict_instance(instance, tree.right)


def calculate_entropy(Y):
    class_counts = {}
    total_samples = len(Y)

    for label in Y:
        if label not in class_counts:
            class_counts[label] = 1
        else:
            class_counts[label] += 1

    entropy = 0
    for count in class_counts.values():
        p = count / total_samples
        entropy -= p * (p and np.log2(p))

    return entropy


def prepare_data(data):
    split_index = int(np.ceil(2/3 * len(data)))
    training_data = data[:split_index]
    validation_data = data[split_index:]

    x_train = training_data.iloc[:, :-1].values
    y_train = training_data.iloc[:, -1].values
    x_valid = validation_data.iloc[:, :-1].values
    y_valid = validation_data.iloc[:, -1].values

    return x_train, x_valid, y_train, y_valid

def probabilities(Y):
    total_samples = len(Y)
    unique_classes, class_counts = np.unique(Y, return_counts=True)
    
    probabilities = {}
    for c, co in zip(unique_classes, class_counts):
        probabilities[c] = co / total_samples
    return probabilities


if __name__ == '__main__':
    ID3()