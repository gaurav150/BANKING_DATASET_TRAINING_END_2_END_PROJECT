import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -------------------------
# Load encoders
# -------------------------
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\label_encoder_contact.pkl", "rb") as f:
    label_encoder_contact = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\label_encoder_housing.pkl", "rb") as f:
    label_encoder_housing = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\label_encoder_default.pkl", "rb") as f:
    label_encoder_default = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\label_encoder_loan.pkl", "rb") as f:
    label_encoder_loan = pickle.load(f)

with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\one_hot_encoder_job.pkl", "rb") as f:
    onehot_encoder_job = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\one_hot_encoder_marital.pkl", "rb") as f:
    onehot_encoder_marital = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\one_hot_encoder_education.pkl", "rb") as f:
    onehot_encoder_education = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\one_hot_encoder_day_of_week.pkl", "rb") as f:
    onehot_encoder_week = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\one_hot_encoder_month.pkl", "rb") as f:
    onehot_encoder_month = pickle.load(f)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\one_hot_encoder_poutcome.pkl", "rb") as f:
    onehot_encoder_poutcome = pickle.load(f)

# Load training columns (saved earlier from x_train DataFrame)
with open(r"D:\Kris NAIK NLP JULY\Impact_of_Digital_Banking_Adoption_On_Customer\columns_name.pkl", "rb") as f:
    training_columns = pickle.load(f)

# -------------------------
# ANN model builder
# -------------------------
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_dim=input_dim))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -------------------------
# Helper: One-hot to DataFrame
# -------------------------
def onehot_to_df(encoder, series, prefix):
    arr = encoder.transform(series).toarray()
    cols = [f"{prefix}_{cat}" for cat in encoder.categories_[0]]
    return pd.DataFrame(arr, columns=cols)

# -------------------------
# Streamlit UI
# -------------------------
st.title("💳 Customer Term Deposit Prediction (ANN Model)")
st.write("Enter client details to predict the likelihood of term deposit subscription.")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", onehot_encoder_job.categories_[0])
marital = st.selectbox("Marital Status", onehot_encoder_marital.categories_[0])
education = st.selectbox("Education", onehot_encoder_education.categories_[0])
default = st.selectbox("Default", label_encoder_default.classes_)
balance = st.number_input("Balance", value=1000)
housing = st.selectbox("Housing Loan", label_encoder_housing.classes_)
loan = st.selectbox("Personal Loan", label_encoder_loan.classes_)
contact = st.selectbox("Contact Method", label_encoder_contact.classes_)
day_of_week = st.selectbox("Day of Week", onehot_encoder_week.categories_[0])
month = st.selectbox("Month", onehot_encoder_month.categories_[0])
duration = st.number_input("Duration (seconds)", value=120)
campaign = st.number_input("Number of contacts during campaign", value=1)
pdays = st.number_input("Days since last contact", value=-1)
previous = st.number_input("Number of contacts before campaign", value=0)
poutcome = st.selectbox("Previous Outcome", onehot_encoder_poutcome.categories_[0])

# Create dataframe
input_data = pd.DataFrame([{
    "age": age, "job": job, "marital": marital, "education": education,
    "default": default, "balance": balance, "housing": housing, "loan": loan,
    "contact": contact, "day_of_week": day_of_week, "month": month,
    "duration": duration, "campaign": campaign, "pdays": pdays,
    "previous": previous, "poutcome": poutcome
}])

# -------------------------
# Preprocessing
# -------------------------

# Label encoders
input_data["contact"] = label_encoder_contact.transform(input_data["contact"])
input_data["housing"] = label_encoder_housing.transform(input_data["housing"])
input_data["default"] = label_encoder_default.transform(input_data["default"])
input_data["loan"] = label_encoder_loan.transform(input_data["loan"])

# One-hot encoders
encoded_job = onehot_to_df(onehot_encoder_job, input_data[["job"]], "job")
encoded_marital = onehot_to_df(onehot_encoder_marital, input_data[["marital"]], "marital")
encoded_education = onehot_to_df(onehot_encoder_education, input_data[["education"]], "education")
encoded_week = onehot_to_df(onehot_encoder_week, input_data[["day_of_week"]], "day_of_week")
encoded_month = onehot_to_df(onehot_encoder_month, input_data[["month"]], "month")
encoded_poutcome = onehot_to_df(onehot_encoder_poutcome, input_data[["poutcome"]], "poutcome")

# Final dataframe
input_data = pd.concat([
    input_data.drop(columns=["job","marital","education","day_of_week","month","poutcome"]),
    encoded_job, encoded_marital, encoded_education,
    encoded_week, encoded_month, encoded_poutcome
], axis=1)

# -------------------------
# Reindex to match training columns (53 features)
# -------------------------
input_data = input_data.reindex(columns=training_columns, fill_value=0)

# Convert to numpy
input_data = np.array(input_data).astype(np.float32)

# -------------------------
# Load Model + Predict
# -------------------------
input_dim = input_data.shape[1]
if input_dim != len(training_columns):
    st.error(f"Input shape mismatch! Got {input_dim}, expected {len(training_columns)}.")
    st.stop()

model = build_model(input_dim)
model.load_weights("ann_model.h5")

if st.button("Predict"):
    prediction = model.predict(input_data)
    prob = prediction[0][0]

    if prob < 0.5:
        st.error(f"❌ The client is unlikely to subscribe. (Prob={prob:.2f})")
    else:
        st.success(f"✅ The client is likely to subscribe. (Prob={prob:.2f})")
