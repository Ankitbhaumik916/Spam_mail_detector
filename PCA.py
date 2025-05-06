import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Only using this from sklearn

# 1. Load and preprocess dataset
data = pd.read_csv('Mall_Customers.csv')
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values
y = np.where(X[:, 1] > 50, 1, -1)  # High spender = 1, Low = -1

# 2. PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 3. Custom Linear SVM Class
class SimpleSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)

# 4. Train the model
split = int(0.7 * len(X_pca))
X_train, X_test = X_pca[:split], X_pca[split:]
y_train, y_test = y[:split], y[split:]

svm = SimpleSVM()
svm.fit(X_train, y_train)

# 5. Accuracy
y_pred = svm.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy (Custom SVM + PCA): {accuracy * 100:.2f}%")

# 6. Visualization
def plot_decision_boundary(X, y, model):
    def decision_func(x1, x2):
        return -(model.w[0]*x1 + model.b)/model.w[1]

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=30)
    
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = decision_func(x_vals, 0)
    plt.plot(x_vals, y_vals, 'k--', label="SVM Decision Boundary")
    
    plt.title("Custom SVM with PCA (2D)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_decision_boundary(X_pca, y, svm)
