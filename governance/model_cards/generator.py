"""
Automated Model Card Generator
Enterprise MLOps Platform

Generates standardized documentation for every trained model.
No model enters production without a model card.
"""

import json
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


client = MlflowClient()


def generate_model_card(run_id: str, model_name: str = "fraud-detection-model") -> dict:
    """Generate a model card from an MLflow run."""

    run = client.get_run(run_id)
    params = run.data.params
    metrics = run.data.metrics

    card = {
        "model_card_version": "1.0",
        "generated_at": datetime.now().isoformat(),

        "model_details": {
            "name": model_name,
            "run_id": run_id,
            "framework": "XGBoost",
            "type": "Binary Classification — Fraud Detection",
            "parameters": params,
        },

        "intended_use": {
            "primary": "Real-time transaction fraud detection",
            "users": "Fraud detection platform, automated scoring pipeline",
            "out_of_scope": "Credit decisioning, customer segmentation, marketing",
        },

        "training_data": {
            "dataset": params.get("dataset", "unknown"),
            "features": int(params.get("num_features", 0)),
            "class_balance": f"Fraud rate ~0.4% (imbalanced)",
            "split_method": "Time-based (80/20) — no future data leakage",
        },

        "performance": {
            "auc_pr": metrics.get("auc_pr"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1_score": metrics.get("f1_score"),
            "true_positives": int(metrics.get("true_positives", 0)),
            "false_positives": int(metrics.get("false_positives", 0)),
            "false_negatives": int(metrics.get("false_negatives", 0)),
            "true_negatives": int(metrics.get("true_negatives", 0)),
        },

        "ethical_considerations": {
            "bias_testing": "pending",
            "fairness_metrics": "pending — required before production",
            "demographic_analysis": "pending",
        },

        "limitations": [
            "Trained on synthetic data — production deployment requires validation on real transactions",
            "Feature store latency assumptions based on local testing, not production infrastructure",
            "Fraud patterns limited to 5 typologies — real-world fraud is more diverse",
            "Geographic coverage limited to UK, US, India — model may underperform in other regions",
        ],

        "governance": {
            "regulatory_review": "pending",
            "bias_check_passed": False,
            "explainability_check_passed": False,
            "approved_for_production": False,
            "approved_by": None,
            "approval_date": None,
        },
    }

    return card


def save_model_card(card: dict, output_path: str = None):
    """Save model card as JSON."""
    if output_path is None:
        output_path = f"governance/model_cards/card_{card['model_details']['run_id'][:8]}.json"

    with open(output_path, 'w') as f:
        json.dump(card, f, indent=2)

    print(f"Model card saved: {output_path}")
    return output_path


def print_model_card(card: dict):
    """Print model card in readable format."""
    print(f"\n{'='*60}")
    print(f"MODEL CARD — {card['model_details']['name']}")
    print(f"{'='*60}")
    print(f"Generated: {card['generated_at']}")
    print(f"Run ID: {card['model_details']['run_id'][:8]}...")
    print(f"Framework: {card['model_details']['framework']}")

    print(f"\nINTENDED USE")
    print(f"  Primary: {card['intended_use']['primary']}")
    print(f"  Out of scope: {card['intended_use']['out_of_scope']}")

    print(f"\nTRAINING DATA")
    print(f"  Dataset: {card['training_data']['dataset']}")
    print(f"  Features: {card['training_data']['features']}")
    print(f"  Split: {card['training_data']['split_method']}")

    perf = card['performance']
    print(f"\nPERFORMANCE")
    print(f"  AUC-PR:    {perf['auc_pr']:.4f}")
    print(f"  Precision: {perf['precision']:.4f}")
    print(f"  Recall:    {perf['recall']:.4f}")
    print(f"  F1:        {perf['f1_score']:.4f}")
    print(f"  Caught: {perf['true_positives']} | Missed: {perf['false_negatives']} | False alarms: {perf['false_positives']}")

    print(f"\nETHICAL CONSIDERATIONS")
    for k, v in card['ethical_considerations'].items():
        print(f"  {k}: {v}")

    print(f"\nLIMITATIONS")
    for lim in card['limitations']:
        print(f"  - {lim}")

    gov = card['governance']
    print(f"\nGOVERNANCE STATUS")
    print(f"  Bias check:          {'PASSED' if gov['bias_check_passed'] else 'PENDING'}")
    print(f"  Explainability:      {'PASSED' if gov['explainability_check_passed'] else 'PENDING'}")
    print(f"  Production approved: {'YES' if gov['approved_for_production'] else 'NO'}")


if __name__ == "__main__":
    experiment = mlflow.get_experiment_by_name("fraud-detection-synthetic")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.auc_pr DESC"])
    best_run_id = runs.iloc[0]['run_id']

    card = generate_model_card(best_run_id)
    print_model_card(card)
    save_model_card(card)
