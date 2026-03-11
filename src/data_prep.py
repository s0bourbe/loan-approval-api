import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_clean_data(filepath):
    print("Loading and cleaning data...")
    df = pd.read_csv(filepath)

    # Drop rows with missing values
    df = df.dropna().reset_index(drop=True)

    # Basic sanity limits
    df = df[df['person_age'] <= 100]
    df = df[df['person_emp_length'] <= 60]

    return df

def preprocess_and_split(df):
    print("Preprocessing data...")

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']   # 0 = good, 1 = default

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = [
        'person_home_ownership',
        'loan_intent',
        'loan_grade',
        'cb_person_default_on_file'
    ]
    numerical_cols = [
        'person_age',
        'person_income',
        'person_emp_length',
        'loan_amnt',
        'loan_int_rate',
        'loan_percent_income',
        'cb_person_cred_hist_length'
    ]

    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_test[col] = le.transform(X_test[col])
        encoders[col] = le

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)

    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

    print("Data preparation complete!")

if __name__ == "__main__":
    df = load_and_clean_data('data/raw/credit_risk_dataset.csv')
    preprocess_and_split(df)
