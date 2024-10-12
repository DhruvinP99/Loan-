import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load your dataset
df = pd.read_csv('E:\LUC Fall 2024\ML\HW_3\Loan.csv')  # Replace with your dataset path

# Handle missing values
df.dropna(inplace=True)

# Convert categorical variables to numerical (if any)
df = pd.get_dummies(df)

# Split features and target variable
X = df.drop('loan_status', axis=1)  # Adjust 'target' to your actual target column name
y = df['loan_status']

# Split the data into training (70%) and development (30%) sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_dev = scaler.transform(X_dev)
X_test = scaler.transform(X_test)

# SVM
model_svm = SVC()
model_svm.fit(X_train, y_train)

# Evaluate on the development set
y_dev_pred_svm = model_svm.predict(X_dev)
print("SVM Classification Report:")
print(classification_report(y_dev, y_dev_pred_svm))

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid_svm = {'C': [0.01, 0.1, 1, 10, 100],}
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_search_svm.fit(X_train, y_train)

# Evaluate the best model
best_svm = grid_search_svm.best_estimator_
y_test_pred_svm = best_svm.predict(X_test)
print("Tuned SVM Classification Report:")
print(classification_report(y_test, y_test_pred_svm))
