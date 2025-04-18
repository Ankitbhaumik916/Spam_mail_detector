# 📧 Spam Mail Detection using Single-Layer Perceptron (SLP)

Welcome to the Spam Mail Detection project — a machine learning model built using a **Single-Layer Perceptron (SLP)** to classify emails as spam or ham (not spam). This lightweight model demonstrates the power of basic neural networks in solving real-world classification problems efficiently.

---

## 🔍 Project Overview

Email spam is a persistent problem in digital communication. This project focuses on detecting spam emails using a **Single-Layer Perceptron**, a type of linear binary classifier. It's a great entry point to understanding how neural networks work on text data.

### 🧠 Model Used
- **SLP (Single-Layer Perceptron)**: A simple yet effective linear classifier trained using the **perceptron learning rule**.

---

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Libraries**:
  - `sklearn` – for dataset splitting and evaluation
  - `numpy` – for numerical operations
  - `nltk` – for text preprocessing
  - `matplotlib` – for optional visualizations (accuracy/loss curve)

---

## 📂 Project Structure

```
📆SpamDetection_SLP/
 ├ 📄 slp_spam_detector.py     ← Main script for training and prediction
 ├ 📄 preprocess.py            ← Email cleaning, tokenization, and vectorization
 ├ 📄 spam_dataset.csv         ← Sample labeled dataset
 ├ 📄 README.md                ← You're reading it!
```

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/spam-slp-detector.git
cd spam-slp-detector
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Model

```bash
python slp_spam_detector.py
```

The script will:
- Preprocess the text data
- Convert text to binary features
- Train a perceptron on the training set
- Evaluate on the test set with accuracy, precision, recall, F1-score

---

## 🧪 Sample Output

```
Training Accuracy: 96.4%
Test Accuracy: 95.2%
F1-Score: 0.947
```

---

## 🧱 Core Concepts Used

- **Perceptron Rule** (weight update: `w = w + α(x * y)`)
- **Bag of Words** for feature extraction
- **Text preprocessing**: tokenization, stopword removal, vectorization
- **Binary classification**: Spam vs. Not Spam

---

## 🌟 What’s More to Come

Here's what’s planned for future releases of this project:

| Feature                            | Status       |
|------------------------------------|--------------|
| ✅ SLP implementation               | Complete     |
| 🔄 Multi-layer Perceptron upgrade  | Coming Soon  |
| 🔍 TF-IDF vectorization            | Coming Soon  |
| 🧠 Integration with sklearn's Perceptron | Coming Soon  |
| 📊 Interactive Streamlit dashboard | Planned      |
| 📱 Flutter mobile app for email checking | In Progress  |
| 🔐 NLP-based phishing detection     | Future Scope |
| ☁️ Deployment via Flask or FastAPI  | Planned      |

---

## 🤝 Contributing

Got a cool feature or improvement in mind? Feel free to fork the repo and send a pull request. All contributions are welcome!

---

## 🧑‍💻 Made by

**Ankit Bhaumik**  
_B.Tech AI Engineering Student_  
💡 Passionate about AI, CV, and product design.  

---

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.
