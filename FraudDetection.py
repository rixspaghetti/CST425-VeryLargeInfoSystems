# Step 1: Import Libraries
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Create a synthetic dataset
data = pd.DataFrame({
    "transaction_amount": [5000, 50, 3000, 20, 2000, 7000, 15, 1200, 10000, 5, 4500, 30, 2500, 700, 8000],
    "location": ["New", "Old", "New", "Old", "Old", "New", "Old", "Old", "New", "Old", "New", "Old", "New", "Old", "New"],
    "frequency": ["High", "Low", "Low", "Low", "High", "High", "Low", "Low", "High", "Low", "High", "Low", "High", "Low", "High"],
    "device": ["New", "Old", "New", "Old", "Old", "New", "Old", "New", "New", "Old", "New", "Old", "New", "Old", "New"],
    "past_fraud": ["Yes", "No", "No", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes"],
    "transaction_time": ["Night", "Morning", "Afternoon", "Night", "Morning", "Night", "Afternoon", "Morning", "Night", "Morning",
                         "Afternoon", "Night", "Afternoon", "Morning", "Night"],
    "merchant_type": ["Online", "In-store", "Online", "ATM", "In-store", "Online", "ATM", "In-store", "Online", "ATM",
                      "Online", "ATM", "Online", "In-store", "Online"],
    "fraud": ["Yes", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No", "No", "No", "Yes", "No", "Yes"]
})


# Step 3: Encode Categorical Variables
label_encoder = {}
for col in ["location", "frequency", "device", "past_fraud", "transaction_time", "merchant_type"]:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoder[col] = le

# Step 4: Scale the Numerical Features
scaler = StandardScaler()
data["tansaction_amount"] = scaler.fit_transform(data[["transaction_amount"]])

# Step 5: Split the data into training and testing sets
X = data.drop("fraud", axis=1) # features
y = data["fraud"] # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train and compare Naive Bayes models
models = {
    "Gaussian Naive Bayes": GaussianNB(),
    "Bernoulli Naive Bayes": BernoulliNB()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model: {name} Accuracy: {accuracy:.2f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("="*50)




