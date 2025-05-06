import pandas as pd
import numpy as np

# 1. Load Dataset
data = pd.read_csv('Mall_Customers.csv')
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# 2. Basic fake labels: Split based on spending score (simulate clustering)
y = np.where(X[:, 1] > 50, 1, 0)  # High vs low spenders

# 3. Custom SVM
class SimpleSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(X.shape[1])
        self.b = 0

        for _ in range(self.n_iters):
            for i, x_i in enumerate(X):
                condition = y_[i] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[i]))
                    self.b -= self.lr * y_[i]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

# 4. Train/Test Split (manual)
split = int(0.7 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. Train and Evaluate
svm = SimpleSVM()
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
acc = np.mean(y_pred == np.where(y_test <= 0, -1, 1))
print(f"Accuracy: {acc * 100:.2f}%")
import matplotlib.pyplot as plt

# Predicted labels converted to 0 and 1
y_pred_binary = np.where(y_pred == -1, 0, 1)

# 1. Scatter plot
plt.figure(figsize=(8, 6))
plt.title('Custom SVM Classification + Decision Boundary')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

# Test predictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_binary, cmap='coolwarm', edgecolor='k', s=70, label='Test Predictions')

# Optional: Training points
plt.scatter(X_train[:, 0], X_train[:, 1],
            c=np.where(y_train == 0, 'gray', 'green'), marker='x', alpha=0.4, label='Training Data')

# 2. Decision Boundary Line: w.x - b = 0 → x2 = (b + w1*x1)/w2
w = svm.w
b = svm.b
x_vals = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_vals = (b + w[0] * x_vals) / w[1]

plt.plot(x_vals, y_vals, 'k--', linewidth=2, label='SVM Decision Boundary')

plt.legend()
plt.grid(True)
plt.show()
