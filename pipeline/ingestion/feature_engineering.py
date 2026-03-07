"""
Feature Engineering Pipeline for Fraud Detection
Enterprise MLOps Platform

This module transforms raw transaction data into ML-ready features.
In production, this same code runs in both:
  - Batch mode (for model training)
  - Real-time mode (for transaction-time scoring via feature store)

Keeping feature logic in one place prevents training-serving skew.
"""

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw transaction data into ML-ready features.

    Args:
        df: Raw transaction DataFrame with columns: Time, V1-V28, Amount, Class

    Returns:
        DataFrame with original + engineered features
    """
    features = df.copy()

    # Time-based features
    features['hour'] = (features['Time'] / 3600) % 24
    features['is_night'] = ((features['hour'] >= 0) & (features['hour'] < 6)).astype(int)

    # Amount-based features
    features['amount_log'] = np.log1p(features['Amount'])
    features['amount_zscore'] = (features['Amount'] - features['Amount'].mean()) / features['Amount'].std()
    features['amount_bin'] = pd.cut(
        features['Amount'],
        bins=[0, 1, 10, 50, 200, 1000, float('inf')],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)
    features['is_round_amount'] = (features['Amount'] % 10 == 0).astype(int)

    # Interaction features
    features['v14_x_v12'] = features['V14'] * features['V12']
    features['v17_x_amount'] = features['V17'] * features['amount_log']
    features['v14_x_amount'] = features['V14'] * features['amount_log']
    features['v14_v11_ratio'] = features['V14'] / (features['V11'] + 1e-6)

    # Aggregated risk score
    features['fraud_risk_signal'] = -(features['V17'] + features['V14'] + features['V12'] + features['V10'])

    return features

def get_feature_columns(df: pd.DataFrame) -> list:
    """Return list of feature columns (excluding target and raw time)."""
    return [col for col in df.columns if col not in ['Class', 'Time']]

def prepare_train_test(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Time-based train/test split.

    Why time-based, not random?
    In production, models only see past data. Random splitting
    leaks future information into training and gives unrealistic metrics.
    """
    df_featured = engineer_features(df)
    feature_cols = get_feature_columns(df_featured)

    split_index = int(len(df_featured) * train_ratio)

    X_train = df_featured[feature_cols].iloc[:split_index]
    X_test = df_featured[feature_cols].iloc[split_index:]
    y_train = df_featured['Class'].iloc[:split_index]
    y_test = df_featured['Class'].iloc[split_index:]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Quick test
    df = pd.read_csv('data/raw/creditcard.csv')
    X_train, X_test, y_train, y_test = prepare_train_test(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train fraud rate: {y_train.mean()*100:.3f}%")
    print(f"Test fraud rate: {y_test.mean()*100:.3f}%")
