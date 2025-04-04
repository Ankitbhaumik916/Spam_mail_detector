import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset from local storage
file_path = "spam_ham_dataset.csv" # Update with your actual file path
data = pd.read_csv(file_path)

# Display first few rows
print("Dataset Loaded Successfully!\n", data.head())

# Keep only necessary columns
data = data[['label', 'text']]  
data.columns = ['Label', 'Message']

# Encode labels (Spam = 1, Not Spam = 0)
encoder = LabelEncoder()
data['Label'] = encoder.fit_transform(data['Label'])

# Convert text to numerical features using Bag-of-Words
vectorizer = CountVectorizer(binary=True, stop_words='english')
X = vectorizer.fit_transform(data['Message']).toarray()
y = data['Label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single Layer Perceptron Model
class SingleLayerPerceptron:
    def __init__(self, input_size, lr=0.01, epochs=100):
        self.weights = np.zeros(input_size + 1)  # +1 for bias
        self.lr = lr
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0  # Step function

    def train(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Adding bias term
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred
                self.weights += self.lr * error * x_i  # Update rule

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1)  # Adding bias term
            predictions.append(self.activation(np.dot(self.weights, x_i)))
        return np.array(predictions)

# Train perceptron
perceptron = SingleLayerPerceptron(input_size=X_train.shape[1], lr=0.01, epochs=10)
perceptron.train(X_train, y_train)

# Test perceptron
y_pred = perceptron.predict(X_test)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
