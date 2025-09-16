import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print("Import pd and np completed")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print("Import StandardScaler completed")
import tensorflow as tf
from pathlib import Path

input = Path(__file__).resolve().parents[2] / 'data' / 'processed'
df = pd.read_csv(input / 'processed_samples.csv')

print("File loaded")
train_df = df[df["run_id"] == "R001"]
test_df = df[df["run_id"] == "R002"]

features = ['alpha_deg']
target = ['CL']

print("Loading data")
X_train = train_df[features].values
y_train = train_df['CL'].values
X_test = test_df[features].values
y_test = test_df['CL'].values
print("Data loaded")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data transformed")
model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("Model compiled")
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
print("Training complete")
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")
