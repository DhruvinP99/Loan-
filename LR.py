import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

# Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

# Evaluate on the development set
y_dev_pred_lr = model_lr.predict(X_dev)
print("Logistic Regression Classification Report:")
print(classification_report(y_dev, y_dev_pred_lr))

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search_lr = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search_lr.fit(X_train, y_train)

# Evaluate the best model
best_lr = grid_search_lr.best_estimator_
y_test_pred_lr = best_lr.predict(X_test)
print("Tuned Logistic Regression Classification Report:")
print(classification_report(y_test, y_test_pred_lr))
