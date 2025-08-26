import streamlit as st  
import joblib
import numpy as np

# Load your trained model
model = joblib.load("rf_tuned.pkl")

# Title
st.title("📊 Customer Churn Prediction App")

st.write("Enter customer details below to predict whether they will churn or not.")

# Input form
with st.form("churn_form"):
    Account_Length = st.number_input("Account Length", min_value=0, max_value=300, value=100)
    Area_Code = st.selectbox("Area Code", [408, 415, 510])
    International_Plan = st.selectbox("International Plan", ["No", "Yes"])
    Voice_Mail_Plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
    Voice_Mail_Messages = st.number_input("Voice Mail Messages", min_value=0, max_value=60, value=0)
    Total_Day_Minutes = st.number_input("Total Day Minutes", min_value=0.0, value=180.0)
    Total_Day_Calls = st.number_input("Total Day Calls", min_value=0, value=100)
    Total_Day_Charge = st.number_input("Total Day Charge", min_value=0.0, value=30.0)
    Total_Evening_Minutes = st.number_input("Total Evening Minutes", min_value=0.0, value=200.0)
    Total_Evening_Calls = st.number_input("Total Evening Calls", min_value=0, value=100)
    Total_Evening_Charge = st.number_input("Total Evening Charge", min_value=0.0, value=20.0)
    Total_Night_Minutes = st.number_input("Total Night Minutes", min_value=0.0, value=200.0)
    Total_Night_Calls = st.number_input("Total Night Calls", min_value=0, value=100)
    Total_Night_Charge = st.number_input("Total Night Charge", min_value=0.0, value=20.0)
    Total_International_Minutes = st.number_input("Total International Minutes", min_value=0.0, value=10.0)
    Total_International_Calls = st.number_input("Total International Calls", min_value=0, value=5)
    Total_International_Charge = st.number_input("Total International Charge", min_value=0.0, value=3.0)
    Customer_Service_Calls = st.number_input("Customer Service Calls", min_value=0, value=1)
    Total_Minutes = st.number_input("Total Minutes", min_value=0.0, value=600.0)
    Total_Calls = st.number_input("Total Calls", min_value=0, value=300)
    Total_Charges = st.number_input("Total Charges", min_value=0.0, value=100.0)
    Avg_Call_Duration = st.number_input("Average Call Duration", min_value=0.0, value=3.0)
    Customer_Service_Category = st.selectbox("Customer Service Category", ["Low", "Medium", "High"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Apply label encoding manually (must match training)
    Int_Plan = 1 if International_Plan == "Yes" else 0
    VM_Plan = 1 if Voice_Mail_Plan == "Yes" else 0
    Cust_Service_Map = {"Low": 0, "Medium": 1, "High": 2}
    Cust_Service_Cat = Cust_Service_Map[Customer_Service_Category]

    # Create feature vector (⚠️ must match training order)
    features = np.array([
        Account_Length, Area_Code, 0,  # Phone_Number not used
        Int_Plan, VM_Plan, Voice_Mail_Messages,
        Total_Day_Minutes, Total_Day_Calls, Total_Day_Charge,
        Total_Evening_Minutes, Total_Evening_Calls, Total_Evening_Charge,
        Total_Night_Minutes, Total_Night_Calls, Total_Night_Charge,
        Total_International_Minutes, Total_International_Calls, Total_International_Charge,
        Customer_Service_Calls,
        Total_Minutes, Total_Calls, Total_Charges, Avg_Call_Duration, Cust_Service_Cat
    ]).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f"⚠️ This customer is likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ This customer is NOT likely to churn (Probability: {prob:.2f})")
