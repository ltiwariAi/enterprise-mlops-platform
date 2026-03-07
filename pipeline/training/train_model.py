"""
Model Training Pipeline for Fraud Detection
Enterprise MLOps Platform

Trains XGBoost classifier with MLflow experiment tracking.
Every training run is logged: parameters, metrics, model artifacts.
"""

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    f1_score, confusion_matrix
)
import sys
sys.path.insert(0, '.')
from pipeline.ingestion.feature_engineering import prepare_train_test


def train_fraud_model(
    data_path: str = "data/raw/creditcard.csv",
    experiment_name: str = "fraud-detection",
    run_name: str = "default_run",
    params: dict = None
):
    """
    Train a fraud detection model with full MLflow tracking.

    Args:
        data_path: Path to raw transaction data
        experiment_name: MLflow experiment name
        run_name: Name for this training run
        params: XGBoost hyperparameters (uses defaults if None)

    Returns:
        dict with model, metrics, and MLflow run_id
    """
    # Load and prepare data using production feature engineering
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    # Calculate class weight for imbalanced data
    scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    # Default parameters — our best performing config
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "scale_pos_weight": scale,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1
        }

    # Train with MLflow tracking
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc_pr = average_precision_score(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn)

        metrics = {
            "auc_pr": auc_pr,
            "roc_auc": roc_auc,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn
        }

        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"Run: {run_name}")
        print(f"AUC-PR: {auc_pr:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"Caught: {tp} | Missed: {fn} | False alarms: {fp}")
        print(f"MLflow run ID: {run.info.run_id}")

        return {
            "model": model,
            "metrics": metrics,
            "run_id": run.info.run_id,
            "X_test": X_test,
            "y_test": y_test,
            "y_prob": y_prob
        }


if __name__ == "__main__":
    result = train_fraud_model(run_name="production_candidate")
    print(f"\nModel trained and tracked successfully.")
    print(f"View at: mlflow ui --port 5001")
