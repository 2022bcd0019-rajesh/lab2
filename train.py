import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


NAME = "SIBYKANNA"
ROLL_NO = "2022BCD0039"


data = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = data.drop("quality", axis=1)
y = data["quality"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"Name: {NAME}")
print(f"Roll No: {ROLL_NO}")
print(f"MSE: {mse}")
print(f"R2 Score: {r2}")


os.makedirs("output/model", exist_ok=True)
os.makedirs("output/results", exist_ok=True)

joblib.dump(model, "output/model/model.pkl")

results = {
    "name": NAME,
    "roll_no": ROLL_NO,
    "mse": mse,
    "r2_score": r2
}

with open("output/results/results.json", "w") as f:
    json.dump(results, f, indent=4)




