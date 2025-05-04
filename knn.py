import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# Step 1: Load and clean the dataset
df = pd.read_csv('adult.csv', na_values=' ?')  # Automatically marks " ?" as NaN
df.dropna(inplace=True)  # Drop rows with missing values

# Step 2: One-hot encode categorical columns
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'native-country']
df = pd.get_dummies(df, columns=categorical_cols)

# Step 3: Convert target to binary (0: <=50K, 1: >50K)
df['income'] = df['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# Step 4: Split features and labels
X = df.drop('income', axis=1)
y = df['income']

# Step 5: Normalize numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 6: Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 7: Use smaller subsets for debugging (you can scale up later)
X_train_small = X_train[:500]
y_train_small = y_train.iloc[:500].to_numpy()
X_test_small = X_test[:10]
y_test_small = y_test.iloc[:10].to_numpy()

# Step 8: Custom KNN Class
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _dist(self, x1, x2):
        return np.linalg.norm(x1 - x2)

    def _predict(self, x):
        distances = [self._dist(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

# Step 9: Run and evaluate
knn = KNN(k=3)
knn.fit(X_train_small, y_train_small)
y_pred = knn.predict(X_test_small)

print("Predictions:", y_pred)
print("Actual     :", y_test_small)
print("\nConfusion Matrix:\n", confusion_matrix(y_test_small, y_pred))
print("\nClassification Report:\n", classification_report(y_test_small, y_pred))
