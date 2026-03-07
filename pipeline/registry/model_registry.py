"""
Model Registry Management
Enterprise MLOps Platform
"""

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
MODEL_NAME = "fraud-detection-model"


def register_model(run_id: str, description: str, trained_by: str = "system"):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri, MODEL_NAME)

    client.update_model_version(name=MODEL_NAME, version=result.version, description=description)
    client.set_model_version_tag(MODEL_NAME, result.version, "trained_by", trained_by)
    client.set_model_version_tag(MODEL_NAME, result.version, "regulatory_review", "pending")

    print(f"Registered {MODEL_NAME} v{result.version}")
    return result.version


def promote_to_staging(version: int):
    client.transition_model_version_stage(MODEL_NAME, version, "Staging")
    print(f"v{version} → Staging")


def promote_to_production(version: int):
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.current_stage == "Production":
            client.transition_model_version_stage(MODEL_NAME, int(v.version), "Archived")
            print(f"v{v.version} → Archived (was Production)")

    client.transition_model_version_stage(MODEL_NAME, version, "Production")
    print(f"v{version} → Production")


def rollback():
    archived = [
        v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.current_stage == "Archived"
    ]
    if not archived:
        print("No archived model to rollback to.")
        return

    latest_archived = sorted(archived, key=lambda v: int(v.version), reverse=True)[0]

    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.current_stage == "Production":
            client.transition_model_version_stage(MODEL_NAME, int(v.version), "Archived")

    client.transition_model_version_stage(MODEL_NAME, int(latest_archived.version), "Production")
    print(f"Rolled back to v{latest_archived.version}")


def get_production_model():
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        if v.current_stage == "Production":
            model = mlflow.xgboost.load_model(f"models:/{MODEL_NAME}/{v.version}")
            print(f"Loaded production model v{v.version}")
            return model, int(v.version)
    print("No production model found.")
    return None, None


def compare_versions(v1: int, v2: int):
    ver1 = client.get_model_version(MODEL_NAME, v1)
    ver2 = client.get_model_version(MODEL_NAME, v2)
    run1 = client.get_run(ver1.run_id)
    run2 = client.get_run(ver2.run_id)

    print(f"\nv{v1} ({ver1.current_stage}) vs v{v2} ({ver2.current_stage})")
    print("-" * 50)
    for metric in ["auc_pr", "precision", "recall"]:
        val1 = run1.data.metrics.get(metric, 0)
        val2 = run2.data.metrics.get(metric, 0)
        winner = f"v{v1}" if val1 >= val2 else f"v{v2}"
        print(f"  {metric:<15} v{v1}: {val1:.4f}  v{v2}: {val2:.4f}  → {winner}")


if __name__ == "__main__":
    print(f"Registry: {MODEL_NAME}")
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"  v{v.version}: {v.current_stage}")
