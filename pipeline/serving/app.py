"""
Model Serving API for Fraud Detection
Enterprise MLOps Platform
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import numpy as np
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from datetime import datetime

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud scoring for transaction data",
    version="1.0.0"
)

# Schemas
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float
    Time: float = 0.0

class PredictionResponse(BaseModel):
    fraud_probability: float
    risk_tier: str
    is_fraud: bool
    model_version: int
    prediction_time_ms: float
    top_features: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[int]
    uptime_seconds: float
    total_predictions: int

# State
MODEL_NAME = "fraud-detection-model"
model = None
model_version = None
start_time = datetime.now()
prediction_count = 0
prediction_log = []


def load_model():
    global model, model_version
    client = MlflowClient()

    for stage in ["Production", "Staging"]:
        versions = [
            v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
            if v.current_stage == stage
        ]
        if versions:
            latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
            model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{latest.version}")
            model_version = int(latest.version)
            print(f"Loaded {MODEL_NAME} v{model_version} ({stage})")
            return

    print("WARNING: No model found in registry")


def engineer_serving_features(tx: dict) -> np.ndarray:
    """Same feature logic as training pipeline — prevents training-serving skew."""
    amount = tx['Amount']
    time_val = tx['Time']

    hour = (time_val / 3600) % 24
    is_night = 1 if 0 <= hour < 6 else 0
    amount_log = np.log1p(amount)
    amount_zscore = (amount - 88.29) / 250.11

    if amount <= 1: amount_bin = 0
    elif amount <= 10: amount_bin = 1
    elif amount <= 50: amount_bin = 2
    elif amount <= 200: amount_bin = 3
    elif amount <= 1000: amount_bin = 4
    else: amount_bin = 5

    is_round = 1 if amount % 10 == 0 else 0
    v14_x_v12 = tx['V14'] * tx['V12']
    v17_x_amount = tx['V17'] * amount_log
    v14_x_amount = tx['V14'] * amount_log
    v14_v11_ratio = tx['V14'] / (tx['V11'] + 1e-6)
    fraud_risk_signal = -(tx['V17'] + tx['V14'] + tx['V12'] + tx['V10'])

    raw_features = [tx[f'V{i}'] for i in range(1, 29)]
    raw_features.append(amount)
    engineered = [hour, is_night, amount_log, amount_zscore, amount_bin,
                  is_round, v14_x_v12, v17_x_amount, v14_x_amount,
                  v14_v11_ratio, fraud_risk_signal]

    return np.array(raw_features + engineered).reshape(1, -1)


@app.on_event("startup")
def startup():
    load_model()


@app.get("/health", response_model=HealthResponse)
def health_check():
    uptime = (datetime.now() - start_time).total_seconds()
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        model_version=model_version,
        uptime_seconds=round(uptime, 2),
        total_predictions=prediction_count
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    global prediction_count

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pred_start = datetime.now()

    tx_dict = transaction.dict()
    features = engineer_serving_features(tx_dict)
    fraud_prob = float(model.predict_proba(features)[0][1])
    is_fraud = fraud_prob >= 0.5

    if fraud_prob >= 0.8: risk_tier = "critical"
    elif fraud_prob >= 0.5: risk_tier = "high"
    elif fraud_prob >= 0.3: risk_tier = "medium"
    else: risk_tier = "low"

    feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount'] + [
        'hour', 'is_night', 'amount_log', 'amount_zscore', 'amount_bin',
        'is_round_amount', 'v14_x_v12', 'v17_x_amount', 'v14_x_amount',
        'v14_v11_ratio', 'fraud_risk_signal'
    ]
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[-5:][::-1]
    top_features = [
        {"feature": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in top_idx
    ]

    pred_time = (datetime.now() - pred_start).total_seconds() * 1000

    prediction_count += 1
    prediction_log.append({
        "timestamp": datetime.now().isoformat(),
        "fraud_probability": fraud_prob,
        "risk_tier": risk_tier,
        "model_version": model_version,
        "latency_ms": round(pred_time, 2)
    })

    return PredictionResponse(
        fraud_probability=round(fraud_prob, 6),
        risk_tier=risk_tier,
        is_fraud=is_fraud,
        model_version=model_version,
        prediction_time_ms=round(pred_time, 2),
        top_features=top_features
    )


@app.get("/predictions/recent")
def recent_predictions():
    return {"predictions": prediction_log[-20:], "total": prediction_count}
