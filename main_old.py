import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline as SKPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
#  step 1 - Loading the dataset 
data = pd.read_csv("TransactionData.csv")
df = data.copy()
correlation = df.select_dtypes(include=["number"]).corr()
# step 2 - Stratified Shuffle Split on the basis of isFraud column
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(df,df["isFraud"]):
    strat_train_set = df.iloc[train_index]
    strat_test_set = df.iloc[test_index]
# step 3 - making  copy of the training data
fraud = strat_train_set.copy()
X_test = strat_test_set.drop("isFraud",axis=1)
X_label = strat_test_set["isFraud"].copy()
# step 4 - separating Features and Labels
features = fraud.drop("isFraud", axis=1)
labels = fraud["isFraud"].copy()
# step 5 - differntiating numerical and categorical attributes
num_attributes = features.select_dtypes(include = ["number"]).columns.tolist()
cat_attributes = ["type"]
# step 6 - making numerical pipeline
num_pipeline = SKPipeline([
               ("impute",SimpleImputer(strategy="median")),
               ("scaler",StandardScaler()) 
])
# step 7 - making categorical pipeline
cat_pipeline = SKPipeline ([
               ("encoder",OneHotEncoder(handle_unknown="ignore"))
])
# step 8 - making coulmn transformer 
preprocessor = ColumnTransformer([
               ("numeric",num_pipeline,num_attributes),
               ("categorical",cat_pipeline,cat_attributes)
])
# step 9 - Applying SMOTE (Synthetic Minority Oversampling Technique)
final_pipeline = ImbPipeline([
                  ("preprocess",preprocessor),
                  ("smote",SMOTE(random_state=42)),
                  ("model",DecisionTreeClassifier(class_weight="balanced"))
                  
])
final_pipeline.fit(features,labels)
X_result = final_pipeline.predict(X_test)
accuracy = accuracy_score(X_result,X_label)
recall = recall_score(X_result,X_label)
f1 = f1_score(X_result,X_label)
precision = precision_score(X_result,X_label)
print(accuracy)
print(recall)
print(f1)
print(precision)
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(X_result, X_label))
print(classification_report(X_result, X_label))

# # step 10 - Trainig the model 
# scores_1 = cross_validate(
#     final_pipeline,
#     features,
#     labels,
#     scoring=["accuracy","precision","recall","f1","roc_auc"],
#     cv=2
# )
# print("Cross validation completed!")
# print(scores_1.keys())
# print("Accuracy :", scores_1["test_accuracy"].mean())
# print("Precision :", scores_1["test_precision"].mean())
# print("Recall   :", scores_1["test_recall"].mean())
# print("F1 Score :", scores_1["test_f1"].mean())
# print("ROC AUC  :", scores_1["test_roc_auc"].mean())
# # So the recommended model will be XGBoost 
# 
