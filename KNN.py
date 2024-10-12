# knn_classifier.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from collections import Counter

# Load your dataset
df = pd.read_csv('E:\LUC Fall 2024\ML\HW_3\Loan.csv')  # Replace with the correct path
df.dropna(inplace=True)

# Convert categorical variables to numerical (if any)
df = pd.get_dummies(df)

# Split features and target variable
X = df.drop('loan_status', axis=1)  # Replace 'target' with your actual target column name
y = df['loan_status']

# Split the data into training, development, and test sets (70%, 15%, 15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# KNN classifier from scratch
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)  # Convert to NumPy array
        self.y_train = np.array(y_train)  # Convert to NumPy array
    
    def predict(self, X_dev):
        predictions = [self._predict(x) for x in X_dev]
        return np.array(predictions)

    def _predict(self, x):
        # Vectorized distance computation
        distances = np.linalg.norm(self.X_train - x, axis=1)
        # Get the indices of the k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k-nearest neighbors
        k_nearest_labels = self.y_train[k_indices]
        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Tune k for best performance
best_k = 0
best_score = 0

for k in range(1, 11):
    knn_model = KNN(k=k)
    knn_model.fit(X_train, y_train)
    y_dev_pred = knn_model.predict(X_dev)
    score = np.mean(y_dev_pred == y_dev)
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best k value: {best_k}")

# Use best k value to evaluate on development set
knn_model = KNN(k=best_k)
knn_model.fit(X_train, y_train)
y_dev_pred = knn_model.predict(X_dev)
print("KNN (Best k) Classification Report:")
print(classification_report(y_dev, y_dev_pred))

# Evaluation metrics for KNN
print("Accuracy (KNN):", accuracy_score(y_dev, y_dev_pred))
print("F1 Score (KNN):", f1_score(y_dev, y_dev_pred, average='weighted'))
print("Confusion Matrix (KNN):\n", confusion_matrix(y_dev, y_dev_pred))

# Baseline using DummyClassifier
dummy_clf = DummyClassifier(strategy='most_frequent')
dummy_clf.fit(X_train, y_train)
y_dummy_pred = dummy_clf.predict(X_dev)
print("Dummy Classifier Report:")
print(classification_report(y_dev, y_dummy_pred))

