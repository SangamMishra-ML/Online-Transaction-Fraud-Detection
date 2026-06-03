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
nameDest = st.text_input("Name of Person Receiving Transaction(In case of Transfers)", value="XXXXXXXXXXXX")
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
   
    if prediction == 0 and newbalanceOrig == (oldbalanceOrg - amount):
        st.success("Your Transaction is a Legitimate Transaction ! Good to go ✅")

    elif prediction == 0:
        st.error("Value Mismatch. Please check the values")

    else:
        st.error("Your Transaction seems to be Fraud. Please Re-check!! 🚨")
with st.sidebar:
    st.header("Model Metrics")
    st.write("Training Model")
    st.write("Random Forest Classifier")
    st.write("Accuracy","-"," 0.9997")
    st.write("Precison","-"," 0.9824")
    st.write("F1 score","-"," 0.8730")
    st.write("Recall Score","-"," 0.7869")
    st.write("ROC AUC score","-"," 0.9756")
with st.expander("Guidelines to use the Model"):
     st.write("You can check the model by performing following Test Transactions.")
     st.write ("step = 15",
            "type = CASH_OUT",
            "amount = 10000",
           " oldbalanceOrg = 50000",
           " newbalanceOrig = 40000",
           " oldbalanceDest = 100000",
            "newbalanceDest = 110000")
     st.write("step = 30"
            "type = TRANSFER",
            "amount = 1000000",
            "oldbalanceOrg = 1000000",
            "newbalanceOrig = 0",
            "oldbalanceDest = 0",
            "newbalanceDest = 0")