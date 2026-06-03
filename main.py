import os
import joblib
import pandas as pd 
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline as SKPipeline 
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as IMBPipeline 

PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes,cat_attributes):

    num_pipeline = SKPipeline([
                   ("impute",SimpleImputer(strategy="median")),
                   ("scaler", StandardScaler())
    ])

    cat_pipeline = SKPipeline([
                ("encoder",OneHotEncoder())
    ])
    
    preprocessor = ColumnTransformer([
        ("numerical",num_pipeline,num_attributes),
        ("categorical",cat_pipeline,cat_attributes)
    ])
    
    final_pipeline = SKPipeline([
        ("preprocess",preprocessor),
        # ("smote",SMOTE(random_state=42)),
        ("model",RandomForestClassifier(class_weight="balanced"))
    ])

    return final_pipeline

if not os.path.exists(PIPELINE_FILE):
    # It's time for model training 
    data = pd.read_csv("TransactionData.csv")
    # making copy of the original data 
    df = data.copy()
    # create the stratified Shuffle Split
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index,test_index in split.split(df,df["isFraud"]):
        df.iloc[test_index].to_csv("input.csv",index=False)
        fraud_data = df.iloc[train_index]
    
    features = fraud_data.drop("isFraud",axis=1)
    labels = fraud_data["isFraud"].copy()

    num_attributes = features.select_dtypes(include = ["number"]).columns.tolist()
    cat_attributes = ["type"]

    pipeline = build_pipeline(num_attributes,cat_attributes)
    pipeline.fit(features,labels)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model is trained , Congrats!!")

else :
    # lets"s do the inference
    pipeline = joblib.load(PIPELINE_FILE)
    input_data = pd.read_csv("input.csv")
    predicted_data = pipeline.predict(input_data)
    input_data["isFraud predicted"] = predicted_data
    input_data.to_csv("output.csv", index = False)
    print()
    print("Inference is complete, results saved to output.csv Enjoy!")

