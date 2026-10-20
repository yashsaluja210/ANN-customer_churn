import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv("customer.csv")

# Data cleaning and handling missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop("customerID", axis=1)

# Handle internet service 'no internet service' values
names = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
for cols in names:
    df[cols] = df[cols].replace('No internet service', 'No')

df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

# Map binary categorical columns to numbers
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"]
for colss in binary_cols:
    if colss == "gender":
        df[colss] = df[colss].map({'Female': 1, 'Male': 0})
    else:
        df[colss] = df[colss].map({'Yes': 1, 'No': 0})

# Feature engineering
service_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 'StreamingTV', 'StreamingMovies']
df["TotalServices"] = df[service_cols].sum(axis=1)

df["Tenuregroup"] = pd.cut(df["tenure"], bins=[0, 12, 24, 48, 72], labels=["New", "1-2 years", "2-4 years", "4-6 years"])
df["Tenuregroup"] = df["Tenuregroup"].map({"New": 0, "1-2 years": 1, "2-4 years": 2, "4-6 years": 3})

# One-hot encoding categorical variables
df = pd.get_dummies(data=df, columns=["Contract", "PaymentMethod", "InternetService", "Tenuregroup"], dtype="int")

# Scale numerical features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
nums_cols = ["tenure", "MonthlyCharges", "TotalCharges", "TotalServices"]
df[nums_cols] = scaler.fit_transform(df[nums_cols])

# Split data into features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Train-test split
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Build ANN model
model = Sequential()
model.add(Flatten(input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

# Fit model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(acc)

# Save artifacts
import joblib
joblib.dump(scaler, "scaler2.pkl")
model.save("customer_churn_ann_model.keras")
joblib.dump(list(X.columns), "columns2.pkl")
