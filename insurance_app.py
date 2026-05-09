import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Title
st.title("Insurance Prediction App")

# Read Dataset
df = pd.read_csv("insurance.csv")

st.write(df.head())

# Encoding
le = LabelEncoder()

for col in ['sex', 'smoker', 'region']:
    df[col] = le.fit_transform(df[col])

# Features and Target
X = df.drop('charges', axis=1)
y = df['charges']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# User Inputs
age = st.slider("Age", 18, 70, 30)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

bmi = st.slider("BMI", 10.0, 50.0, 25.0)

children = st.slider("Children", 0, 5, 0)

smoker = st.selectbox(
    "Smoker",
    ["yes", "no"]
)

region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

# Encoding Inputs
sex_encoded = 1 if sex == "male" else 0
smoker_encoded = 1 if smoker == "yes" else 0

region_mapping = {
    "southwest": 0,
    "southeast": 1,
    "northwest": 2,
    "northeast": 3
}

region_encoded = region_mapping[region]

input_data = np.array([[
    age,
    sex_encoded,
    bmi,
    children,
    smoker_encoded,
    region_encoded
]])

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)

    st.success(
        f"Predicted Insurance Cost: ${prediction[0]:,.2f}"
    )
