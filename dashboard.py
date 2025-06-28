import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# Title of the dashboard
st.title("COVID-19 Dashboard")

# Load the dataset
df = pd.read_csv("data/covid_19_data.csv")

# Convert the 'Observation_date' column to datetime format
df['Observation_date'] = pd.to_datetime(df['Observation_date'])

# Sidebar filters
st.sidebar.header("Filter Data")
selected_country = st.sidebar.selectbox("Select a Country", sorted(df['Country_Region'].unique()))

# Convert pandas Timestamps to Python datetime.date for slider
min_date = df['Observation_date'].min().date()
max_date = df['Observation_date'].max().date()

# Date range slider
date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date)
)

# Filter data based on selections
filtered_df = df[
    (df['Country_Region'] == selected_country) &
    (df['Observation_date'].dt.date >= date_range[0]) &
    (df['Observation_date'].dt.date <= date_range[1])
]

# Show summary metrics
st.subheader(f"Summary for {selected_country}")
st.metric("Total Confirmed", int(filtered_df['Confirmed'].sum()))
st.metric("Total Deaths", int(filtered_df['Deaths'].sum()))
st.metric("Total Recovered", int(filtered_df['Recovered'].sum()))

# Line chart of confirmed cases over time
st.subheader("Confirmed Cases Over Time")
fig, ax = plt.subplots()
ax.plot(filtered_df['Observation_date'], filtered_df['Confirmed'], color='blue', marker='o')
ax.set_xlabel("Date")
ax.set_ylabel("Confirmed Cases")
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("Predict Next Day's Confirmed Cases")

# Prepare data for prediction
filtered_df = filtered_df.sort_values('Observation_date')
filtered_df = filtered_df.drop_duplicates(subset='Observation_date')  # Avoid duplicate dates

filtered_df['Days'] = (filtered_df['Observation_date'] - filtered_df['Observation_date'].min()).dt.days
X = filtered_df['Days'].values.reshape(-1, 1)
y = filtered_df['Confirmed'].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for next day
next_day = filtered_df['Days'].max() + 1
predicted_cases = model.predict([[next_day]])

# Show prediction
st.success(f"Predicted Confirmed Cases for Next Day: {int(predicted_cases[0])}")

# ----------------------------------------
# Additional Visualization: Deaths Over Time
# ----------------------------------------
st.subheader("Deaths Over Time")

fig2, ax2 = plt.subplots()
ax2.bar(filtered_df['Observation_date'], filtered_df['Deaths'], color='red')
ax2.set_xlabel("Date")
ax2.set_ylabel("Deaths")
ax2.tick_params(axis='x', rotation=45)
st.pyplot(fig2)