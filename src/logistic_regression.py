import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression():
    def __init__(self, learning_rate=0.1, epochs=10000, small_value=1e-15):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.small_value = small_value
        self.weights = None

    def init_weights(self, num_features):
        return np.random.uniform(-1e-4, 1e-4, size=num_features)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, h):
        epsilon = self.small_value
        return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

    def run_logistic_regression(self, X_train, Y_train):
        num_samples, num_features = X_train.shape
        X_train = np.column_stack((X_train, np.ones(num_samples))) 
        mean_logs = []
        self.weights = self.init_weights(num_features + 1)

        for curr_epoch in range(self.epochs):
            z = np.dot(X_train, self.weights)
            h = self.sigmoid(z)
            
            gradient = np.dot(X_train.T, (h - Y_train)) / num_samples
            self.weights -= self.learning_rate * gradient

            log_loss_training = self.log_loss(Y_train, h)
            mean_logs.append(log_loss_training)

            if curr_epoch > 0 and abs(mean_logs[curr_epoch - 1] - mean_logs[curr_epoch]) < self.small_value:
                break

        return self.weights, mean_logs

    def predict(self, X):
        X = np.column_stack((X, np.ones(X.shape[0])))
        z = np.dot(X, self.weights)
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)



