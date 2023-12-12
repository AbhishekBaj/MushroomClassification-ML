from statistics import mode
import numpy as np

class Node:
    def __init__(self, value=None, feature=None, left=None, right=None, probabilities=None):
        self.value = value
        self.feature = feature
        self.left = left
        self.right = right
        self.probabilities = probabilities

class DecisionTree():
    def __init__(self):
        pass

    def myDT(self, x_train, y_train, x_valid):
        root = self.DTL(x_train, y_train, set(range(x_train.shape[1])), mode(y_train))
        y_valid_pred = self.traverse_tree(x_valid, root)
        return y_valid_pred

    def DTL(self, X, Y, features, default):
        if len(X) == 0:
            return Node(value=default)
        elif np.unique(Y).size == 1:
            return Node(value=Y[0])
        elif len(features) == 0:
            return Node(value=default, probabilities= self.probabilities(Y))
        else:
            best = self.choose_attribute(X, Y, list(features) )
            tree = Node(feature=best)

            X_true, Y_true = X[X[:, best] == 1], Y[X[:, best] == 1]
            left_child = self.DTL(X_true, Y_true, features - {best}, mode(Y))
            tree.left = left_child

            X_false, Y_false = X[X[:, best] == 0], Y[X[:, best] == 0]
            right_child = self.DTL(X_false, Y_false, features - {best}, mode(Y))
            tree.right = right_child
            return tree

    def calculate_entropy(self, Y):
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

    def choose_attribute(self, X, Y, feature_list):
        best_attribute = None
        best_entropy = float('inf')

        for feature in feature_list:
            entropy = self.calculate_entropy(Y[X[:, feature] == 1])
            if entropy < best_entropy:
                best_entropy = entropy
                best_attribute = feature
        return best_attribute

    def predict_instance(self, instance, tree):
        if tree is None:
            return 0
        if tree.value:
            return tree.value

        if tree.feature and instance[tree.feature]:
            return self.predict_instance(instance, tree.left)
        else:
            return self.predict_instance(instance, tree.right)

    def traverse_tree(self, X, tree):
        predictions = []
        for instance in X:
            predictions.append(self.predict_instance(instance, tree))
        predictions = np.array(predictions, dtype=object)
        return predictions


    def probabilities(self, Y):
        total_samples = len(Y)
        unique_classes, class_counts = np.unique(Y, return_counts=True)
        
        probabilities = {}
        for c, co in zip(unique_classes, class_counts):
            probabilities[c] = co / total_samples
        return probabilities