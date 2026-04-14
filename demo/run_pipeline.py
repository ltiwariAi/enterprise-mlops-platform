"""
Enterprise MLOps Platform — Simulated Pipeline Run
---------------------------------------------------

Walks through every stage of the fraud-detection MLOps lifecycle with
synthesized-but-realistic output. Uses only the Python stdlib so it runs
anywhere, including sandboxed environments without pandas/xgboost/mlflow.

Stages:
  1. Data ingestion & feature engineering
  2. Model training (XGBoost + MLflow)
  3. Model registry (versioning)
  4. Real-time serving (cold start + scoring)
  5. Drift monitoring (PSI)
  6. Automated retraining (champion vs challenger)
  7. Governance gates + production promotion

Run:
    python3 demo/run_pipeline.py
    python3 demo/run_pipeline.py --fast    # no sleep pacing
"""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
import time
from datetime import datetime, timedelta


# ----------------------------- terminal helpers -----------------------------


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


PACE = 0.4  # seconds between major prints; overridden by --fast


def pause(mult: float = 1.0) -> None:
    time.sleep(PACE * mult)


def rule(char: str = "=", width: int = 72) -> str:
    return char * width


def header(title: str) -> None:
    print()
    print(C.BOLD + C.BLUE + rule("=") + C.RESET)
    print(C.BOLD + C.BLUE + f"  {title}" + C.RESET)
    print(C.BOLD + C.BLUE + rule("=") + C.RESET)


def stage(n: int, total: int, title: str) -> None:
    print()
    print(C.BOLD + C.CYAN + f"[STAGE {n}/{total}] {title}" + C.RESET)
    print(C.DIM + rule("-") + C.RESET)


def log(msg: str, level: str = "INFO") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    colors = {
        "INFO": C.GRAY,
        "OK": C.GREEN,
        "WARN": C.YELLOW,
        "ERR": C.RED,
        "STEP": C.CYAN,
    }
    color = colors.get(level, C.GRAY)
    print(f"{C.DIM}{ts}{C.RESET} {color}{level:<4}{C.RESET} {msg}")


def ok(msg: str) -> None:
    log(msg, "OK")


def warn(msg: str) -> None:
    log(msg, "WARN")


def step(msg: str) -> None:
    log(msg, "STEP")


def kv(pairs: list[tuple[str, str]], indent: int = 4) -> None:
    width = max(len(k) for k, _ in pairs)
    pad = " " * indent
    for k, v in pairs:
        print(f"{pad}{C.DIM}{k.ljust(width)}{C.RESET}  {v}")


def table(rows: list[list[str]], headers: list[str]) -> None:
    all_rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]
    fmt = "  " + "  ".join("{:<" + str(w) + "}" for w in widths)
    print(C.BOLD + fmt.format(*headers) + C.RESET)
    print("  " + "  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(x) for x in r]))


# ----------------------------- fake state -----------------------------------


random.seed(42)

RUN_ID = hashlib.sha1(f"run-{datetime.now().isoformat()}".encode()).hexdigest()[:12]
MODEL_NAME = "fraud-detection-model"
DATASET_ROWS = 284_807
FRAUD_ROWS = 492  # ~0.1727%


# ----------------------------- stage implementations ------------------------


def stage_1_ingestion() -> dict:
    stage(1, 7, "Data Ingestion & Feature Engineering")
    step("Loading raw transactions from data/raw/creditcard.csv ...")
    pause()
    log(f"Loaded {DATASET_ROWS:,} transactions ({FRAUD_ROWS} fraud, "
        f"{FRAUD_ROWS / DATASET_ROWS:.4%} base rate)")
    pause(0.5)

    step("Running feature engineering pipeline ...")
    features = [
        "amount_log", "hour_sin", "hour_cos", "day_of_week",
        "amt_vs_mean_24h", "txn_count_1h", "merchant_risk",
        "v14_x_amount", "v10_x_v14", "v4_x_v12", "is_weekend",
    ]
    for f in features:
        log(f"  engineered feature: {f}", "INFO")
        time.sleep(PACE * 0.08)
    ok(f"{len(features)} features engineered")
    pause(0.5)

    step("Time-based train/test split (no random split — reflects production) ...")
    train_n = int(DATASET_ROWS * 0.8)
    test_n = DATASET_ROWS - train_n
    kv([
        ("train rows", f"{train_n:,}  (t < 2023-12-01)"),
        ("test rows",  f"{test_n:,}  (t >= 2023-12-01)"),
        ("train fraud", f"{int(FRAUD_ROWS * 0.78)}"),
        ("test fraud",  f"{FRAUD_ROWS - int(FRAUD_ROWS * 0.78)}"),
    ])
    ok("Ingestion complete")
    return {"features": features, "train_n": train_n, "test_n": test_n}


def stage_2_training(ing: dict) -> dict:
    stage(2, 7, "Model Training (XGBoost + MLflow)")
    mlflow_run_id = hashlib.sha1(f"run-{RUN_ID}".encode()).hexdigest()[:16]
    step(f"MLflow run started  run_id={mlflow_run_id}")
    pause(0.3)

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "scale_pos_weight": 577.87,  # ~(neg/pos) for class imbalance
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "seed": 42,
    }
    step("Logging hyperparameters to MLflow ...")
    for k, v in params.items():
        log(f"  param  {k:<17} = {v}", "INFO")
        time.sleep(PACE * 0.05)

    step("Training XGBoost ...")
    for epoch in (10, 50, 100, 200, 300):
        loss = 0.18 * (1 - epoch / 400) + random.uniform(-0.005, 0.005)
        aucpr = 0.60 + 0.25 * (epoch / 300) + random.uniform(-0.01, 0.01)
        log(f"  [boost {epoch:>3}]  train_logloss={loss:.4f}  val_aucpr={aucpr:.4f}", "INFO")
        time.sleep(PACE * 0.4)
    pause(0.3)

    metrics = {
        "aucpr": 0.8421,
        "auc_roc": 0.9817,
        "precision_at_0.5": 0.8930,
        "recall_at_0.5": 0.7645,
        "f1": 0.8241,
    }
    step("Evaluating on held-out test set ...")
    kv([(k, f"{v:.4f}") for k, v in metrics.items()])
    pause(0.5)

    ok(f"Training complete. Logged to MLflow run {mlflow_run_id}")
    return {"run_id": mlflow_run_id, "metrics": metrics, "params": params}


def stage_3_registry(train: dict) -> dict:
    stage(3, 7, "Model Registry")
    step(f"Registering model  name='{MODEL_NAME}'  source='runs:/{train['run_id']}/model'")
    pause(0.4)
    version = 7  # pretend this is the 7th version
    log(f"  created model version {version}")
    pause(0.3)
    step("Tagging version metadata ...")
    kv([
        ("trained_by",        "mlops-pipeline@enterprise"),
        ("dataset",           "creditcard-2023-12"),
        ("regulatory_review", "pending"),
        ("business_owner",    "fraud-risk-team"),
    ])
    pause(0.3)
    step("Transitioning version -> Staging ...")
    pause(0.3)
    ok(f"fraud-detection-model v{version} now in Staging")
    return {"version": version, "stage": "Staging"}


def stage_4_serving(train: dict, reg: dict) -> None:
    stage(4, 7, "Real-Time Serving API (cold start + scoring)")
    step("Starting FastAPI app (uvicorn) on 0.0.0.0:8000 ...")
    pause(0.4)
    ok("uvicorn started — waiting for model")
    step(f"Loading model from registry: models:/{MODEL_NAME}/Staging ...")
    pause(0.5)
    ok(f"Loaded {MODEL_NAME} v{reg['version']} (Staging)  latency_warmup=312ms")
    step("GET /health")
    log("  200  {\"status\": \"ready\", \"model_version\": 7, \"uptime_s\": 0.31}")
    pause(0.4)

    step("POST /predict  (3 sample transactions)")
    samples = [
        ("TXN-00001", 42.10,   "normal grocery",       0.0042, False),
        ("TXN-00002", 1839.00, "high-amount online",   0.4130, False),
        ("TXN-00003", 2994.11, "velocity + night + intl", 0.9418, True),
    ]
    rows = []
    for tid, amt, note, score, is_fraud in samples:
        verdict = (C.RED + "FRAUD" + C.RESET) if is_fraud else (C.GREEN + "OK" + C.RESET)
        rows.append([tid, f"${amt:,.2f}", note, f"{score:.4f}", verdict])
        time.sleep(PACE * 0.3)
    table(rows, ["txn_id", "amount", "note", "fraud_prob", "verdict"])
    pause(0.5)
    ok("Serving healthy. p50=8ms  p95=24ms  p99=41ms  (simulated)")


def stage_5_monitoring() -> dict:
    stage(5, 7, "Drift Monitoring — week 3 in production")
    step("Loading reference distribution (training data) ...")
    pause(0.3)
    step("Loading current window (last 7 days of production) ...")
    pause(0.3)
    step("Computing PSI per feature ...")
    psi_results = [
        ("amount_log",       0.048, "healthy"),
        ("hour_sin",         0.071, "healthy"),
        ("hour_cos",         0.089, "healthy"),
        ("day_of_week",      0.032, "healthy"),
        ("amt_vs_mean_24h",  0.137, "warning"),
        ("txn_count_1h",     0.204, "critical"),
        ("merchant_risk",    0.291, "critical"),
        ("v14_x_amount",     0.094, "healthy"),
        ("v10_x_v14",        0.068, "healthy"),
        ("v4_x_v12",         0.041, "healthy"),
        ("is_weekend",       0.012, "healthy"),
    ]
    rows = []
    for feat, psi, status in psi_results:
        color = {"healthy": C.GREEN, "warning": C.YELLOW, "critical": C.RED}[status]
        rows.append([feat, f"{psi:.3f}", color + status + C.RESET])
        time.sleep(PACE * 0.1)
    table(rows, ["feature", "psi", "status"])
    pause(0.4)

    critical = [f for f, _, s in psi_results if s == "critical"]
    warn(f"Critical drift on {len(critical)} feature(s): {', '.join(critical)}")
    warn("Recommendation: trigger retraining")
    return {"critical_features": critical, "trigger_retrain": True}


def stage_6_retraining(monitor: dict, champion: dict) -> dict:
    stage(6, 7, "Automated Retraining — drift-triggered")
    step(f"Trigger source: drift on {monitor['critical_features']}")
    pause(0.3)
    step("Assembling new training window (last 60 days + fresh labels) ...")
    pause(0.5)
    log("  new training rows: 341,228   new fraud: 587")
    pause(0.3)

    step("Training challenger model (seed=42) ...")
    for epoch in (50, 150, 300):
        aucpr = 0.60 + 0.30 * (epoch / 300) + random.uniform(-0.01, 0.01)
        log(f"  [boost {epoch:>3}]  val_aucpr={aucpr:.4f}", "INFO")
        time.sleep(PACE * 0.4)

    challenger_metrics = {"aucpr": 0.8643, "recall_at_0.5": 0.7951, "f1": 0.8377}
    champion_metrics = champion["metrics"]

    step("Champion vs challenger comparison ...")
    rows = [
        ["aucpr",         f"{champion_metrics['aucpr']:.4f}",
         f"{challenger_metrics['aucpr']:.4f}",
         f"+{(challenger_metrics['aucpr'] - champion_metrics['aucpr']):.4f}"],
        ["recall_at_0.5", f"{champion_metrics['recall_at_0.5']:.4f}",
         f"{challenger_metrics['recall_at_0.5']:.4f}",
         f"+{(challenger_metrics['recall_at_0.5'] - champion_metrics['recall_at_0.5']):.4f}"],
        ["f1",            f"{champion_metrics['f1']:.4f}",
         f"{challenger_metrics['f1']:.4f}",
         f"+{(challenger_metrics['f1'] - champion_metrics['f1']):.4f}"],
    ]
    table(rows, ["metric", "champion v7", "challenger v8", "delta"])
    pause(0.5)

    improvement = challenger_metrics["aucpr"] - champion_metrics["aucpr"]
    min_improvement = 0.01
    if improvement >= min_improvement:
        ok(f"Challenger beats champion by +{improvement:.4f} "
           f"(min required: +{min_improvement:.4f})")
        ok("Promoting challenger to Staging — governance gates still required for Production")
        return {"version": 8, "metrics": challenger_metrics,
                "promoted_to": "Staging", "improvement": improvement}
    else:
        warn(f"Challenger only +{improvement:.4f} — below threshold, not promoting")
        return {"promoted_to": None}


def stage_7_governance(challenger: dict) -> None:
    stage(7, 7, "Governance Gates — Staging -> Production")
    step(f"Running governance gates for {MODEL_NAME} v{challenger['version']} ...")
    pause(0.4)

    gates = [
        ("performance",
         [("aucpr >= 0.80",        f"{challenger['metrics']['aucpr']:.4f}", True),
          ("recall_at_0.5 >= 0.75", f"{challenger['metrics']['recall_at_0.5']:.4f}", True)]),
        ("bias",
         [("disparate_impact >= 0.80  (merchant_category)", "0.913", True),
          ("equalized_odds_diff <= 0.10 (region)",           "0.042", True)]),
        ("explainability",
         [("SHAP TreeExplainer runs", "ok", True),
          ("top-5 features explainable", "ok", True)]),
        ("model_card",
         [("intended_use",   "documented", True),
          ("limitations",    "documented", True),
          ("training_data",  "documented", True)]),
    ]

    all_pass = True
    for gate_name, checks in gates:
        step(f"Gate: {C.BOLD}{gate_name}{C.RESET}")
        for check, value, passed in checks:
            marker = (C.GREEN + "PASS" + C.RESET) if passed else (C.RED + "FAIL" + C.RESET)
            print(f"    [{marker}]  {check:<45} -> {value}")
            time.sleep(PACE * 0.15)
            if not passed:
                all_pass = False
        pause(0.2)

    print()
    if all_pass:
        ok("All governance gates PASSED")
        step("Writing audit log entry ...")
        audit = {
            "timestamp":  datetime.now().isoformat(timespec="seconds"),
            "model":      f"{MODEL_NAME} v{challenger['version']}",
            "action":     "promote_to_production",
            "gates":      "performance + bias + explainability + model_card",
            "signer":     "mlops-governance@enterprise",
            "run_id":     RUN_ID,
        }
        kv(list(audit.items()))
        pause(0.3)
        step(f"Transitioning {MODEL_NAME} v{challenger['version']}: Staging -> Production")
        step(f"Archiving previous Production (v7)")
        pause(0.3)
        ok(f"Promotion complete. {MODEL_NAME} v{challenger['version']} now serving "
           "live traffic.")
    else:
        warn("One or more gates FAILED — promotion blocked")


# ----------------------------- main -----------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulated MLOps pipeline run")
    parser.add_argument("--fast", action="store_true",
                        help="skip pacing delays (fast run for CI)")
    args = parser.parse_args()

    global PACE
    if args.fast:
        PACE = 0.0

    header("ENTERPRISE MLOPS PLATFORM — SIMULATED PIPELINE RUN")
    kv([
        ("run_id",     RUN_ID),
        ("started_at", datetime.now().isoformat(timespec="seconds")),
        ("mode",       "SIMULATION (stdlib only, no real ML deps)"),
        ("model",      MODEL_NAME),
    ])

    ing = stage_1_ingestion()
    train = stage_2_training(ing)
    reg = stage_3_registry(train)
    stage_4_serving(train, reg)
    monitor = stage_5_monitoring()
    challenger = stage_6_retraining(monitor, champion=train)
    if challenger.get("promoted_to") == "Staging":
        stage_7_governance(challenger)
    else:
        warn("Skipping governance stage — no challenger promoted")

    header("PIPELINE RUN COMPLETE")
    kv([
        ("run_id",    RUN_ID),
        ("ended_at",  datetime.now().isoformat(timespec="seconds")),
        ("outcome",   "SUCCESS — v8 promoted to Production"),
    ])
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
