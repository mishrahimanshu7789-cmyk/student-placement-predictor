

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline

# Step 1: Load the dataset
df = pd.read_csv("student.csv")

# Step 2: Remove duplicate and missing records
df = df.drop_duplicates()
df = df.dropna()

# Step 3: Separate input features and target
X = df.drop("Placement", axis=1)
y = df["Placement"]

# Step 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Create scaling + Logistic Regression pipeline
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# Step 6: Train the model
model.fit(X_train, y_train)

# Step 7: Predict on test data
y_pred = model.predict(X_test)

# Step 8: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Placement Prediction Model Trained Successfully!")
print("Total records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Model Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Step 9: Save the trained pipeline
joblib.dump(model, "placement_model.pkl")

# Save the input column names for the GUI
joblib.dump(list(X.columns), "scaler.pkl")

joblib.dump(list(X.columns), "feature_columns.pkl")


print("Feature columns saved as feature_columns.pkl")


print("\nModel saved as placement_model.pkl")
print("Input columns saved as scaler.pkl")
