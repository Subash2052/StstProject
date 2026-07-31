import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load cleaned data
df = pd.read_csv("../data/Churn_Cleaned.csv")

# Independent variables
# Independent variables (optimized model)
X = df[
    [
        "Complains",
        "Frequency of use",
        "Seconds of Use",
        "Charge Amount",
        "Customer Value"
    ]
]

# Dependent variable
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Build model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Regression coefficients
print("Intercept:")
print(model.intercept_[0])

print("\nCoefficients:")
for column, coef in zip(X.columns, model.coef_[0]):
    print(f"{column}: {coef}")

# Accuracy 123
print("\nAccuracy:")
print(accuracy_score(y_test, predictions))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, predictions))