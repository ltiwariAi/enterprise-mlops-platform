"""
Automated Retraining Pipeline
Enterprise MLOps Platform
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix
from datetime import datetime
import sys
sys.path.insert(0, '.')
from pipeline.ingestion.feature_engineering import prepare_train_test
from pipeline.monitoring.drift_detector import DriftDetector


client = MlflowClient()
MODEL_NAME = "fraud-detection-model"


def get_current_production_metrics() -> dict:
    """Get metrics from the current staging/production model."""
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.current_stage in ["Production", "Staging"]:
            run = client.get_run(v.run_id)
            return {
                "version": int(v.version),
                "stage": v.current_stage,
                "auc_pr": run.data.metrics.get("auc_pr", 0),
                "precision": run.data.metrics.get("precision", 0),
                "recall": run.data.metrics.get("recall", 0)
            }
    return None


def retrain(data_path: str = "data/raw/creditcard.csv",
            min_improvement: float = 0.01) -> dict:
    """
    Retrain the model and promote only if it beats the current champion.

    Args:
        data_path: Path to transaction data
        min_improvement: Minimum AUC-PR improvement required to promote

    Returns:
        dict with retrain results
    """
    print(f"\n{'='*60}")
    print(f"AUTOMATED RETRAINING — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Get current champion metrics
    champion = get_current_production_metrics()
    if champion:
        print(f"\nCurrent champion: v{champion['version']} ({champion['stage']})")
        print(f"  AUC-PR: {champion['auc_pr']:.4f}")
    else:
        print("\nNo current champion — first model")

    # Load and prepare data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.05,
        "scale_pos_weight": scale,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "aucpr",
        "random_state": int(datetime.now().timestamp()) % 10000,
        "n_jobs": -1
    }

    # Train with MLflow tracking
    mlflow.set_experiment("fraud-detection")
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\nTraining challenger model...")
        mlflow.log_params(params)
        mlflow.log_param("retrain_trigger", "drift_detected")

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        auc_pr = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn)

        metrics = {
            "auc_pr": auc_pr, "f1_score": f1,
            "precision": precision, "recall": recall,
            "true_positives": tp, "false_positives": fp,
            "false_negatives": fn, "true_negatives": tn
        }
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, "model")

        print(f"  AUC-PR: {auc_pr:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  Caught: {tp} | Missed: {fn} | False alarms: {fp}")

        # Champion-challenger comparison
        result = {
            "run_id": run.info.run_id,
            "run_name": run_name,
            "metrics": metrics,
            "promoted": False,
            "reason": ""
        }

        if champion is None:
            # No champion — auto-promote
            model_uri = f"runs:/{run.info.run_id}/model"
            reg = mlflow.register_model(model_uri, MODEL_NAME)
            client.transition_model_version_stage(MODEL_NAME, int(reg.version), "Staging")
            result["promoted"] = True
            result["reason"] = "first model"
            print(f"\n  PROMOTED to Staging (first model) — v{reg.version}")

        elif auc_pr > champion["auc_pr"] + min_improvement:
            # Challenger wins
            model_uri = f"runs:/{run.info.run_id}/model"
            reg = mlflow.register_model(model_uri, MODEL_NAME)

            # Archive old champion
            for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
                if v.current_stage in ["Production", "Staging"] and int(v.version) != int(reg.version):
                    client.transition_model_version_stage(MODEL_NAME, int(v.version), "Archived")

            client.transition_model_version_stage(MODEL_NAME, int(reg.version), "Staging")
            result["promoted"] = True
            result["reason"] = f"AUC-PR improved by {auc_pr - champion['auc_pr']:.4f}"
            print(f"\n  PROMOTED — v{reg.version}")
            print(f"  Improvement: {champion['auc_pr']:.4f} → {auc_pr:.4f} (+{auc_pr - champion['auc_pr']:.4f})")

        else:
            result["reason"] = f"No improvement (champion: {champion['auc_pr']:.4f}, challenger: {auc_pr:.4f})"
            print(f"\n  NOT PROMOTED — challenger didn't beat champion by {min_improvement}")
            print(f"  Champion: {champion['auc_pr']:.4f} | Challenger: {auc_pr:.4f}")

        return result


def run_pipeline(data_path: str = "data/raw/creditcard.csv"):
    """
    Full pipeline: check drift → retrain if needed → promote if better.
    """
    print("DRIFT CHECK → RETRAIN → EVALUATE → PROMOTE")
    print("=" * 60)

    # Load data and set up drift detector
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = prepare_train_test(df)

    critical_features = ['V14', 'V17', 'V12', 'V10', 'fraud_risk_signal', 'v14_x_v12']
    detector = DriftDetector(X_train, critical_features=critical_features)

    # Check drift on test data
    drift_result = detector.check_drift(X_test, exclude_features=['hour', 'is_night'])

    print(f"\nDrift status: {drift_result['overall_status']}")
    print(f"Features drifted: {len(drift_result['features_drifted'])}")

    if detector.should_retrain():
        print("\nDrift detected — triggering retraining...")
        return retrain(data_path)
    else:
        print("\nNo significant drift — skipping retraining.")
        return {"promoted": False, "reason": "no drift detected"}


if __name__ == "__main__":
    result = retrain()
    print(f"\nResult: {'Promoted' if result['promoted'] else 'Not promoted'} — {result['reason']}")
