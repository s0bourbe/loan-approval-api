# 🏦 Bank Loan Default Predictor

An end-to-end Machine Learning pipeline and web application designed to predict whether an applicant will default on a bank loan. 

This project demonstrates a complete AI engineering workflow: from raw data preprocessing and model training to serving predictions via a REST API and a user-friendly frontend.

![Streamlit Web App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🚀 Live Demo
You can try the web application here: **[Link to your Streamlit App once deployed]**

## 📋 Project Overview
Financial institutions need reliable ways to assess credit risk. This project uses the [Credit Risk Dataset from Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) to build a predictive model.

**Key Features:**
* **Data Engineering:** Handled missing values, removed outliers, scaled numerical data (Income, Age), and encoded categorical data (Home Ownership, Loan Intent).
* **Machine Learning:** Trained a `RandomForestClassifier` optimized for imbalanced classes (`class_weight='balanced'`).
* **Backend API:** Built a scalable REST API using FastAPI to serve the model's predictions.
* **Frontend UI:** Created an interactive Streamlit dashboard allowing users to input applicant details and instantly receive a risk score.

## 🏗️ Repository Structure
```text
loan-approval-api/
│
├── data/
│   ├── raw/                  # Original Kaggle dataset (credit_risk_dataset.csv)
│   └── processed/            # Cleaned and split Train/Test data
│
├── notebooks/                # Jupyter notebooks for Exploratory Data Analysis (EDA)
│
├── src/                      # Source code
│   ├── data_prep.py          # Data cleaning, scaling, and feature engineering
│   └── train.py              # Model training and evaluation
│
├── models/                   # Saved artifacts
│   ├── rf_model.pkl          # Trained Random Forest model
│   ├── scaler.pkl            # StandardScaler for numerical inputs
│   └── encoders.pkl          # LabelEncoders for categorical inputs
│
├── app.py                    # Streamlit Frontend application
├── requirements.txt          # Python dependencies
└── README.md
