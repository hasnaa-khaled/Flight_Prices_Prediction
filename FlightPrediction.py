import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gradio as gr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# ── 1. Download Dataset ──────────────────────────────────────
path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
print("Path to dataset files:", path)

# ── 2. Load Data ─────────────────────────────────────────────
file_path_clean = os.path.join(path, 'Clean_Dataset.csv')
df = pd.read_csv(file_path_clean)
print(df.head())
df.info()
print(df.isnull().sum())

# ── 3. Cleaning ───────────────────────────────────────────────
df = df.drop_duplicates()
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# ── 4. Visualization ──────────────────────────────────────────
sns.set_style("whitegrid")

plt.figure(figsize=(8,5))
sns.histplot(df['price'], kde=True, bins=50)
plt.title("Distribution of Flight Prices")
plt.xlabel("Price")
plt.show()

plt.figure(figsize=(10,6))
sns.boxplot(x='airline', y='price', data=df)
plt.title("Price by Airline")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='stops', y='price', data=df)
plt.title("Price by Number of Stops")
plt.show()

plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x='days_left', y='price', data=df, alpha=0.3)
plt.title("Price vs Days Left Before Departure")
plt.show()

# ── 5. Linear Regression ──────────────────────────────────────
df_no_flight = df.drop(columns=['flight'], errors='ignore')
df_encoded_no_flight = pd.get_dummies(df_no_flight, drop_first=True)

X_no_flight = df_encoded_no_flight.drop('price', axis=1)
y_no_flight = df_encoded_no_flight['price']

X_train_no_flight, X_test_no_flight, y_train_no_flight, y_test_no_flight = train_test_split(
    X_no_flight, y_no_flight, test_size=0.2, random_state=42)

lr_no_flight = LinearRegression()
lr_no_flight.fit(X_train_no_flight, y_train_no_flight)

y_pred_no_flight = lr_no_flight.predict(X_test_no_flight)
rmse_no_flight = np.sqrt(mean_squared_error(y_test_no_flight, y_pred_no_flight))
r2_no_flight = r2_score(y_test_no_flight, y_pred_no_flight)

print(f"Model RMSE: {rmse_no_flight:,.2f}")
print(f"Model R2 Score: {r2_no_flight:.4f}")

# ── 6. Logistic Regression ────────────────────────────────────
threshold = df['price'].median()
df_encoded_no_flight['price_class'] = (df_encoded_no_flight['price'] > threshold).astype(int)

X2 = df_encoded_no_flight.drop(['price', 'price_class'], axis=1)
y2 = df_encoded_no_flight['price_class']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X2_train, y2_train)

y2_pred = log_reg.predict(X2_test)
print("Accuracy:", accuracy_score(y2_test, y2_pred))
print(classification_report(y2_test, y2_pred))


import streamlit as st

# ── Streamlit Interface ───────────────────────────────────────
st.title("✈️ Flight Price Predictor")
st.markdown("Predict the price of a flight based on selected parameters.")

# ── Inputs ────────────────────────────────────────────────────
airline = st.selectbox("Airline", df['airline'].unique().tolist())
source_city = st.selectbox("Source City", df['source_city'].unique().tolist())
departure_time = st.selectbox("Departure Time", df['departure_time'].unique().tolist())
stops = st.selectbox("Stops", df['stops'].unique().tolist())
arrival_time = st.selectbox("Arrival Time", df['arrival_time'].unique().tolist())
destination_city = st.selectbox("Destination City", df['destination_city'].unique().tolist())
flight_class = st.selectbox("Class", df['class'].unique().tolist())
duration = st.number_input("Duration (hours)", min_value=float(df['duration'].min()), max_value=float(df['duration'].max()), value=float(df['duration'].mean()))
days_left = st.number_input("Days Left Before Departure", min_value=int(df['days_left'].min()), max_value=int(df['days_left'].max()), value=int(df['days_left'].mean()))

if st.button("Predict"):
    processed_input = pd.DataFrame(0, index=[0], columns=X_no_flight.columns)
    processed_input['duration'] = duration
    processed_input['days_left'] = days_left

    if f'airline_{airline}' in processed_input.columns:
        processed_input[f'airline_{airline}'] = 1
    if f'source_city_{source_city}' in processed_input.columns:
        processed_input[f'source_city_{source_city}'] = 1
    if f'departure_time_{departure_time}' in processed_input.columns:
        processed_input[f'departure_time_{departure_time}'] = 1
    if f'stops_{stops}' in processed_input.columns:
        processed_input[f'stops_{stops}'] = 1
    if f'arrival_time_{arrival_time}' in processed_input.columns:
        processed_input[f'arrival_time_{arrival_time}'] = 1
    if f'destination_city_{destination_city}' in processed_input.columns:
        processed_input[f'destination_city_{destination_city}'] = 1
    if f'class_{flight_class}' in processed_input.columns:
        processed_input[f'class_{flight_class}'] = 1

    # ── Linear Regression Result ──────────────────────────────
    price_prediction = lr_no_flight.predict(processed_input)[0]
    st.success(f"💰 Predicted Price (Linear Regression): ₹{price_prediction:,.2f}")

    # ── Logistic Regression Result ────────────────────────────
    
    processed_input_log = pd.DataFrame(0, index=[0], columns=X2.columns)
    for col in processed_input.columns:
        if col in processed_input_log.columns:
            processed_input_log[col] = processed_input[col].values

    class_prediction = log_reg.predict(processed_input_log)[0]
    if class_prediction == 1:
        st.warning(" Recommendation (Logistic Regression): This flight is **Expensive** — Book Now before prices rise!")
    else:
        st.info(" Recommendation (Logistic Regression): This flight is **Cheap** — You can Wait for a better deal.")