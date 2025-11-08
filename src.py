# pip install scikit-learn numpy pandas matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("housing_prices.csv")

print("*"*64)
print("Shape (rows, columns):", df.shape, ", Column names:", list(df.columns))
print("First 5 rows:")
print(df.head())

print("\nData types and non-null counts:")
print(df.info())

print("*"*64)
print("Missing values per column:\n", df.isna().sum())

before = len(df)
df = df.drop_duplicates().dropna()
after = len(df)
print(f"Dropped {before - after} rows (duplicates or missing). New size: {df.shape}")

print("*"*64)
print(df.describe())

print("*"*64)
print(df.corr(numeric_only=True))

feature_cols = ["Size_m2", "Bedrooms", "Age_years"]
target_col = "Price"
X, y = df[feature_cols].values, df[target_col].values

print("*"*64)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("*"*64)
print("Train size:", X_train.shape[0], "rows")
print("Test size :", X_test.shape[0], "rows")

model = LinearRegression()
model.fit(X_train, y_train)
print("*"*64)
print("Intercept (b):", model.intercept_)
print("Coefficients (w) mapped to features:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"  {name:>10s}: {coef:.2f}")

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("*"*64)
print("Tests:")
print(f"MSE: {mse:.2f}, R²: {r2:.3f}")

plt.figure()
plt.scatter(df["Size_m2"], df["Price"])
plt.xlabel("Size (m²)")
plt.ylabel("Price")
plt.title("Size vs Price")
plt.show()


plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("True vs Predicted (Test)")
# draw a diagonal reference line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])
plt.show()


residuals = y_test - y_pred
plt.figure()
plt.scatter(y_pred, residuals)
plt.axhline(0)
plt.xlabel("Predicted Price")
plt.ylabel("Residual (True - Pred)")
plt.title("Residuals vs Predicted")
plt.show()
