import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load cleaned data
df = pd.read_csv("../data/Churn_Cleaned.csv")

# Independent variables
X = df[
    [
        "Frequency of use",
        "Seconds of Use",
        "Charge Amount",
        "Subscription Length"
    ]
]

# Dependent variable
y = df["Customer Value"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Print equation
print("Intercept:", model.intercept_)
print()

print("Coefficients:")
for column, coef in zip(X.columns, model.coef_):
    print(f"{column}: {coef}")

print()
print("R-squared:", r2_score(y_test, predictions))