import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data


def z_score_standardization(data, mean, std):
    return (data - mean) / std

def init_weights(num_features):
    return np.random.uniform(-1e-4, 1e-4, size=num_features)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def log_loss(y, h):
    epsilon = 1e-15
    return -np.mean(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))


def run_logistic_regression(X_train, Y_train, epochs, learning_rate, small_value):
    num_samples, num_features = X_train.shape
    X_train = np.column_stack((X_train, np.ones(num_samples))) 
    mean_logs = []
    w = init_weights(num_features + 1)

    for curr_epoch in range(epochs):
        z = np.dot(X_train, w)
        h = sigmoid(z)
        gradient = np.dot(X_train.T, (h - Y_train)) / num_samples
        w -= learning_rate * gradient

        log_loss_training = log_loss(Y_train, h)
        mean_logs.append(log_loss_training)

        if curr_epoch > 0 and abs(mean_logs[curr_epoch - 1] - mean_logs[curr_epoch]) < small_value:
            break

    return w, mean_logs

def classify_samples(X_val, w, threshold=0.5):
    X_val = np.column_stack((X_val, np.ones(X_val.shape[0])))
    z_val = np.dot(X_val, w)
    classification_val = sigmoid(z_val)
    return (classification_val >= threshold).astype(int)

# Randomize the data
np.random.seed(0)
df = preprocess_data()
data = df.sample(frac=1).reset_index(drop=True)

# Split into training and validation
split_index = int(np.ceil(2/3 * len(data)))
training_data = data[:split_index]
validation_data = data[split_index:]

X_train = training_data.iloc[:, :-1].values
Y_train = training_data.iloc[:, -1].values
X_val = validation_data.iloc[:, :-1].values
Y_val = validation_data.iloc[:, -1].values

# Z score the data
mean = X_train.mean()
std = X_train.std()
X_train = z_score_standardization(X_train, mean, std)
mean = X_val.mean()
std = X_val.std()
X_val = z_score_standardization(X_val, mean, std)

# training 
learning_rate = 0.1
epochs = 10500
small_value = 2**(-32)
weights, mean_log_loss_history = run_logistic_regression(X_train, Y_train, epochs, learning_rate, small_value)
predicted_val = classify_samples(X_val, weights)

# Metrics
false_positive = np.sum((Y_val == 0) & (predicted_val == 1))
false_negative = np.sum((Y_val == 1) & (predicted_val == 0))
true_positive = np.sum((Y_val == 1) & (predicted_val == 1))
true_negative = np.sum((Y_val == 0) & (predicted_val == 0))

precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
f1_score = 2 * (precision * recall) / (precision + recall)
accuracy = (true_positive + true_negative) / len(Y_val)

print("precision: ", precision)
print("recall: ", recall)
print("f1_score: ", f1_score)
print("accuracy: ", accuracy)

# Plot
plt.plot(range(len(mean_log_loss_history)), mean_log_loss_history)
plt.ylabel("Log Loss")
plt.xlabel("Epoch")
plt.show()


