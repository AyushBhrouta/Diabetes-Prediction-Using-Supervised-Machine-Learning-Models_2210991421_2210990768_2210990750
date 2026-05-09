# ==========================================
# Data Preprocessing Module
# ==========================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """Load dataset from CSV file"""
    return pd.read_csv(path)


def preprocess_data(df):
    """
    Handle missing values and scale features
    """
    np.random.seed(42)

    # Replace 0 with NaN (as per research paper)
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)

    # Fill missing values using median
    df.fillna(df.median(), inplace=True)

    # Split features and target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y