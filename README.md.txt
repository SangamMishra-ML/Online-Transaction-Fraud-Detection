# Online Transaction Fraud Detection System

## Overview

Online payment fraud is one of the major challenges faced by financial institutions and digital payment platforms. This project aims to detect fraudulent online transactions using Machine Learning techniques. The system analyzes transaction-related features and predicts whether a transaction is legitimate or fraudulent.

The project includes:
- Data preprocessing pipeline
- Exploratory Data Analysis (EDA)
- Handling class imbalance
- Machine Learning model training and evaluation
- Streamlit-based web application for real-time predictions
- End-to-end deployment-ready workflow

---

## Problem Statement

Fraudulent transactions represent a very small percentage of total transactions, making fraud detection a highly imbalanced classification problem.

The objective of this project is to build a robust machine learning model capable of identifying fraudulent transactions while minimizing false negatives and maintaining high recall.

---

## Dataset Information

The dataset contains online transaction records with transaction-related attributes used for fraud detection.

### Features

| Feature | Description |
|----------|------------|
| Amount | Transaction amount |
| Old Balance Origin | Sender's balance before transaction |
| New Balance Origin | Sender's balance after transaction |
| Old Balance Destination | Receiver's balance before transaction |
| New Balance Destination | Receiver's balance after transaction |
| Transaction Type | Type of transaction |
| Is Fraud | Target variable |

> Note: Dataset may contain additional transaction-specific features depending on the source.

---

## Project Workflow

### 1. Data Collection
- Load transaction dataset
- Inspect structure and data types

### 2. Data Preprocessing
- Missing value analysis
- Duplicate removal
- Feature engineering
- Encoding categorical variables
- Scaling numerical features

### 3. Exploratory Data Analysis (EDA)
- Fraud vs Non-Fraud distribution
- Transaction amount analysis
- Correlation analysis
- Fraud patterns visualization
- Class imbalance analysis

### 4. Handling Class Imbalance
Techniques used:
- SMOTE (Synthetic Minority Oversampling Technique)
- Stratified Sampling

### 5. Model Training
Multiple classification algorithms were evaluated:

- Decision Tree
- Random Forest
- XGBoost

### 6. Hyperparameter Tuning
Model performance was improved using:
- Grid Search
- Cross Validation
- Stratified K-Fold Validation

### 7. Model Evaluation
Metrics used:

- Accuracy
- Precision
- Recall
- F1 Score

Since fraud detection is a highly imbalanced problem, Recall and Precision were prioritized over Accuracy.

---

## Model Performance

| Metric | Score |
|----------|---------|
| Accuracy | 0.998% |
| Precision | 0.855% |
| Recall | 0.458% |
| F1 Score | 0.597% |


> Update these values with your final model results.

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Imbalanced-Learn (SMOTE)
- XGBoost
- Joblib

### Deployment
- Streamlit

### Version Control
- Git
- GitHub

---

## Key Learning Outcomes

Through this project, I gained hands-on experience in:

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Handling Imbalanced Datasets
- Machine Learning Model Development
- Hyperparameter Tuning
- Model Evaluation
- Streamlit Deployment
- Git & GitHub Version Control

---

## Author

**Sangam Mishra**

Aspiring Machine Learning, Deep Learning, and Generative AI Engineer

GitHub: https://github.com/SangamMishra-ML/

LinkedIn: https://linkedin.com/in/SangamMishra2006/

---

## License

This project is developed for educational and portfolio purposes.