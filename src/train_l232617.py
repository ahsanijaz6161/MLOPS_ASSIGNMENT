import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression , LogisticRegression


# Load dataset
data = pd.read_csv("data/Housing.csv")

print("Dataset loaded successfully")
print(data)

# Features and target
X = data[["area", "bedrooms", "bathrooms", "stories", "parking"]]
X = X / X.max()
y = data["price"]

# Train model
model = LogisticRegression
#model = LinearRegression()

model.fit(X, y)

# Save model
joblib.dump(model, "model/model.pkl")

print("Model trained successfully")
