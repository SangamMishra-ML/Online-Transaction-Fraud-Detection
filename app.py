import streamlit as st 
import pandas as pd 
import joblib 

# Giving title to the page and setting the configuration og the page 
st.set_page_config(page_title="Online Fraud Detection System", layout="wide")
st.title("Online Transaction Fraud Detection System")
st.subheader("Machine Learning Model trained to detect online Transaction Frauds")
pipeline = joblib.load("pipeline.pkl")
# Taking the input from the users for the prediction 
step = st.number_input("Enter the step of the transaction(It is just a unit of time in hours i.e, 1 Hour = 1 Step)",min_value=1 , max_value=95)
type_ = st.selectbox("Type of Transaction",["PAYMENT","TRANSFER","DEBIT","CREDIT","CASH_OUT","CASH_IN"])
amount = st.number_input("Amount of Transaction",min_value=1.00)
nameOrig = st.text_input("Name of Person Originated Transaction", value="XXXXXXXXXX")
oldbalanceOrg = st.number_input("Old Account Balance of the Person Originated Transaction")
newbalanceOrig = st.number_input("New Account Balance of the Person Originated Transaction")
nameDest = st.text_input("Name of Person Receiving Transaction", value="XXXXXXXXXXXX")
oldbalanceDest = st.number_input("Old Account Balance of the Person Receiving Transaction")
newbalanceDest = st.number_input("New Account Balance of the Person Receiving Transaction")
# making dataframe of the inputs and predicting the result 
if st.button("Predict"):
    input_df = pd.DataFrame ({
               "step" : [step],
               "type" : [type_],
               "amount" : [amount],
               "nameOrig" : [nameOrig],
               "oldbalanceOrg" : [oldbalanceOrg],
               "newbalanceOrig" : [newbalanceOrig],
               "nameDest" : [nameDest],
               "oldbalanceDest" : [oldbalanceDest],
               "newbalanceDest" : [newbalanceDest]
    })

    prediction = pipeline.predict(input_df)
    if prediction == 0 :
        st.success("Your Transaction is a Legitimate Transaction ! Good to go ✅")
    else :
        st.error("Your Transaction seems to be Fraud. Please Re-check!! 🚨")
with st.sidebar:
    st.header("Model Metrics")
    st.write("Training Model")
    st.write("Decison Tree Classifier")
    st.write("Accuracy"," 0.998")
    st.write("Precison"," 0.855")
    st.write("F1 score"," 0.597")
    st.write("Recall Score"," 0.458")