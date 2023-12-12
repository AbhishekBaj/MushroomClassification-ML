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
        np.random.seed(0)
        return np.random.uniform(-1e-4, 1e-4, size=num_features)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def log_loss(self, y, h):
        epsilon = self.small_value
        return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))

    def run_logistic_regression(self, x_train, y_train, x_valid, y_valid):
        
        num_samples, num_features = x_train.shape
        num_samples_v, num_features_v = x_valid.shape
        x_train = np.column_stack((x_train, np.ones(num_samples))) 
        x_valid = np.column_stack((x_valid, np.ones(num_samples_v))) 
        mean_logs = []
        mean_logs_v = []
        self.weights = self.init_weights(num_features + 1)

        for curr_epoch in range(self.epochs):
            z = np.dot(x_train, self.weights)
            h = self.sigmoid(z)
            log_loss_training = self.log_loss(y_train, h)
            mean_logs.append(log_loss_training)

            gradient = np.dot(x_train.T, (h - y_train)) / num_samples
            self.weights -= self.learning_rate * gradient

            z_v = np.dot(x_valid, self.weights)
            y_hat_v = self.sigmoid(z_v)
            lo = self.log_loss(y_valid, y_hat_v)
            mean_logs_v.append(lo)

            if curr_epoch > 0 and abs(mean_logs[curr_epoch - 1] - mean_logs[curr_epoch]) < self.small_value:
                break

        self.create_plot(mean_logs, mean_logs_v)

    def predict(self, X):
        X = np.column_stack((X, np.ones(X.shape[0])))
        z = np.dot(X, self.weights)
        predictions = self.sigmoid(z)
        return (predictions >= 0.5).astype(int)

    def create_plot(self, training, validation):
        plt.plot(range(len(training)), training, label='Training Log Loss')
        plt.plot(range(len(validation)), validation, label='Validation Log Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Log Loss')
        plt.title('Training and Validation Log Loss')
        plt.legend()
        plt.show()


