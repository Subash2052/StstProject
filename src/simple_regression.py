import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
df = pd.read_csv("../data/Churn_Cleaned.csv")
X = df[["Frequency of use"]]

y = df["Customer Value"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Intercept:", model.intercept_)

print("Coefficient:", model.coef_[0])
r2 = r2_score(y_test, predictions)

print("R-squared:", r2)