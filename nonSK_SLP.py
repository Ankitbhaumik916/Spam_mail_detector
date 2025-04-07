import numpy as np
import pandas as pd
import re
import random
from collections import Counter

# Load and prepare dataset
data = pd.read_csv("spam_ham_dataset.csv")[['label', 'text']]
data.columns = ['Label', 'Message']
data['Label'] = data['Label'].apply(lambda x: 1 if x.lower() == 'spam' else 0)

# Preprocess text
def preprocess(text): return re.sub(r'[^a-z\s]', '', text.lower()).split()

# Build vocabulary & BoW
vocab = {word: i for i, word in enumerate(Counter(w for msg in data['Message'] for w in preprocess(msg)))}
def to_bow(msgs):
    X = np.zeros((len(msgs), len(vocab)), dtype=int)
    for i, msg in enumerate(msgs):
        for word in preprocess(msg):
            if word in vocab: X[i][vocab[word]] = 1
    return X

X, y = to_bow(data['Message']), data['Label'].values

# Manual train-test split
idx = list(range(len(X))); random.seed(42); random.shuffle(idx)
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# Perceptron
class Perceptron:
    def __init__(self, size, lr=0.01, epochs=10):
        self.w, self.lr, self.epochs = np.zeros(size + 1), lr, epochs

    def act(self, x): return 1 if x >= 0 else 0

    def train(self, X, y):
        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                xi = np.insert(xi, 0, 1)
                pred = self.act(np.dot(self.w, xi))
                self.w += self.lr * (yi - pred) * xi

    def predict(self, X):
        return np.array([self.act(np.dot(self.w, np.insert(x, 0, 1))) for x in X])

# Train model
model = Perceptron(X_train.shape[1])
model.train(X_train, y_train)
print(f"Accuracy: {np.mean(model.predict(X_test) == y_test) * 100:.2f}%")

# 🔍 Predict user input
def classify_message(msg):
    x = to_bow([msg])[0]
    return "Spam" if model.predict([x])[0] else "Not Spam"

# Take input from user
while True:
    user_input = input("\nEnter a message to classify (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    print("→ Prediction:", classify_message(user_input))
