<!--
Render with Marp:
  npx @marp-team/marp-cli@latest docs/presentation.md --pdf
  npx @marp-team/marp-cli@latest docs/presentation.md --html
  npx @marp-team/marp-cli@latest docs/presentation.md --pptx
-->
---
marp: true
theme: default
paginate: true
size: 16:9
header: "Enterprise MLOps Platform — Fraud Detection"
footer: "© Confidential — Customer Briefing"
style: |
  section { font-family: 'Inter', 'Helvetica', sans-serif; }
  h1 { color: #0b3d91; }
  h2 { color: #0b3d91; border-bottom: 2px solid #0b3d91; padding-bottom: 4px; }
  table { font-size: 0.85em; }
  code { background: #f4f6fb; padding: 2px 6px; border-radius: 4px; }
  .lead { font-size: 1.1em; color: #333; }
  .kpi { font-size: 2.2em; color: #0b3d91; font-weight: 700; }
  .muted { color: #666; font-size: 0.85em; }
---

<!-- _class: lead -->
# Enterprise MLOps Platform
## Real-Time Credit Card Fraud Detection

**A production-grade reference architecture for the full ML lifecycle**

<br>

Customer Briefing · 2026

---

## Executive Summary

- **What it is:** An end-to-end MLOps platform for detecting fraudulent credit card transactions in real time.
- **Why it matters:** Fraud costs the payments industry **$40B+ per year**. Production ML demands more than a good model — it demands the *system* around the model.
- **What we built:** Training, model registry, drift monitoring, automated retraining, governed promotion, and a real-time scoring API — all instrumented and auditable.
- **Status today:** Core ML lifecycle (~**2,000 lines of Python**) is production-grade; deployment infrastructure is the next milestone.

---

## The Business Problem

<div class="lead">

Fraud detection at payment-network scale has four hard constraints:

</div>

| Constraint | Reality |
|---|---|
| **Severe class imbalance** | ~0.17% of transactions are fraud |
| **Latency budget** | < 100 ms end-to-end per transaction |
| **Concept drift** | Fraud patterns evolve weekly as attackers adapt |
| **Regulatory scrutiny** | Every decision must be explainable & auditable |

A "good model in a notebook" is not a fraud-detection system.

---

## Our Solution: Seven Production Capabilities

1. **Feature Engineering Pipeline** — 11 engineered features, time-aware splits
2. **Training Pipeline** — XGBoost + MLflow experiment tracking
3. **Model Registry** — Staging → Production promotion with rollback
4. **Drift Monitoring** — PSI-based feature drift detection
5. **Automated Retraining** — Champion–challenger promotion logic
6. **Real-Time Serving API** — FastAPI scoring service
7. **Governance Layer** — Bias, explainability, audit, and promotion gates

---

## Reference Architecture

```
   ┌────────────┐     ┌──────────────┐      ┌──────────────┐
   │  Raw Data  │────▶│   Feature    │─────▶│   Training   │
   │ (transacts)│     │  Engineering │      │   Pipeline   │
   └────────────┘     └──────────────┘      └──────┬───────┘
                                                   │ log
                                            ┌──────▼───────┐
                                            │   MLflow     │
   ┌────────────┐     ┌──────────────┐      │  Registry    │
   │ Production │◀────│  Governance  │◀─────│ (versioned)  │
   │  Serving   │     │   Gates      │      └──────┬───────┘
   │  (FastAPI) │     └──────────────┘             │
   └─────┬──────┘                                  │ drift?
         │ scores                          ┌───────▼──────┐
         └────────────────────────────────▶│  Monitoring  │
                                           │ + Retraining │
                                           └──────────────┘
```

<span class="muted">Single source of truth for feature engineering ensures no training/serving skew.</span>

---

## Capability 1 — Feature Engineering

**File:** `pipeline/ingestion/feature_engineering.py`

- **11 engineered features** from raw transaction data:
  time-based cyclical features, amount buckets, velocity signals, V*-interactions
- **Time-based train/test split** (not random) — reflects production reality where we train on the past and predict the future
- **Shared between training and serving** — the same code path runs in batch and real time to eliminate training/serving skew, the #1 cause of silent ML failures

---

## Capability 2 — Training Pipeline

**File:** `pipeline/training/train_model.py`

- **XGBoost** classifier with class-weighted loss (`scale_pos_weight`) for the 0.17% fraud base rate
- **MLflow experiment tracking**: parameters, metrics, artifacts, model signature
- **Deterministic** (seed = 42) for reproducibility
- **Business-relevant metrics**: AUC-PR, precision @ recall, confusion matrix — *not just accuracy*
- **Three model variants compared** in `notebooks/02_model_training.ipynb`: baseline, deep, aggressive-recall

---

## Capability 3 — Model Registry

**File:** `pipeline/registry/model_registry.py`

- **MLflow Model Registry** integration
- **Lifecycle:** `None` → `Staging` → `Production` → `Archived`
- **Rollback** with a single call — promote last archived back to Production
- **Metadata tagging:** trained_by, regulatory_review, business_owner
- **Version comparison** — compare any two registered versions on the same eval set

<span class="muted">Every model in Production is versioned, attributable, and reversible.</span>

---

## Capability 4 — Drift Monitoring

**File:** `pipeline/monitoring/drift_detector.py`

- **Population Stability Index (PSI)** per feature — industry standard in financial services
- **Configurable thresholds**: warning (PSI > 0.1), critical (PSI > 0.2)
- **Critical feature watchlist** — drift on key signals triggers immediate retraining
- **Mean-shift tracking** alongside distribution-shift detection
- **Status rollup:** `healthy` / `warning` / `critical` with retraining recommendation

<span class="muted">Catches data drift before it becomes model decay.</span>

---

## Capability 5 — Automated Retraining

**File:** `pipeline/retraining/auto_retrain.py` (216 LOC)

- **Champion–challenger framework** — incumbent must be beaten to be replaced
- **Minimum improvement gate** (default: +1% AUC-PR) — prevents noise-driven promotions
- **Drift-triggered** — retraining fires when monitoring flags critical drift
- **Auto-promotes to Staging** on passing comparison; archives prior version
- **Governance gate** still required to enter Production

<span class="muted">New models only ship when measurably better — no silent regressions.</span>

---

## Capability 6 — Real-Time Serving API

**File:** `pipeline/serving/app.py` (180 LOC)

- **FastAPI** + **Pydantic** strict input validation (`Transaction` schema)
- **OpenAPI docs** at `/docs` — self-documenting for integrators
- **Health endpoint** `/health` with uptime & model status
- **MLflow-backed model loading** with Staging → Production fallback
- **Feature engineering parity** with training — zero skew by construction
- **Graceful degradation:** returns 503 if model not loaded, never a silent failure

---

## Capability 7 — Governance Layer (4 Modules)

**Folder:** `governance/`

| Module | What it does |
|---|---|
| `bias_checker.py` | Disparate impact + equalized odds across protected groups |
| `shap_explainer.py` | Per-prediction SHAP values + global feature importance |
| `generator.py` | Auto-generated **Model Cards** (JSON) from MLflow runs |
| `governance_gate.py` | **4 sequential gates**: performance → bias → explainability → model card |

**No model reaches Production without passing every gate. Every decision is logged.**

---

## What Makes This "Enterprise-Grade"?

<div class="lead">

Five architectural choices that separate a prototype from a platform:

</div>

1. **Time-based data splits** — no lookahead bias; reflects real deployment
2. **Shared feature code (batch + online)** — eliminates training/serving skew
3. **Champion–challenger promotion** — no silent regressions
4. **Governance gates in code** — bias, explainability, and performance enforced programmatically
5. **Model Cards as artifacts** — every production model is documented, attributable, and auditable

---

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Modeling | **XGBoost**, scikit-learn | Core classifier + baselines |
| Tracking & Registry | **MLflow** | Experiments, models, versioning |
| Serving | **FastAPI** + Gunicorn | Real-time scoring API |
| Data | **Pandas**, NumPy | Feature engineering |
| Explainability | **SHAP** (TreeExplainer) | Regulatory transparency |
| Observability (target) | OpenTelemetry, Prometheus | Tracing + metrics |
| Exploration | **Jupyter** (9 notebooks) | Analyst workflow |

<span class="muted">All open-source, no vendor lock-in.</span>

---

## Notebooks: The Analyst Lifecycle

| Notebook | Phase |
|---|---|
| `01_data_exploration` | EDA, fraud patterns, class imbalance |
| `02_model_training` | Three XGBoost variants, MLflow comparison |
| `03_model_registry` | Promotion / archive / rollback workflow |
| `05_monitoring` | PSI drift detection on live data |
| `06_retraining` | Drift-triggered champion/challenger |
| `07–08_synthetic_data` | Synthetic transaction generation (WIP) |
| `09_governance` | End-to-end promotion with gate enforcement |

Analysts onboard in hours, not weeks.

---

## Key Metrics & Guarantees (Today)

<div class="kpi">284K</div>

transactions modelled — real Kaggle credit-card fraud dataset

<br>

<div class="kpi">0.17%</div>

fraud base rate — handled via class-weighted learning

<br>

<div class="kpi">AUC-PR</div>

primary metric — correct choice for severe class imbalance

---

## Governance & Compliance Readiness

- **Auditability:** every promotion decision logged with timestamp, reason, and signer
- **Explainability:** SHAP values available per prediction on request
- **Bias monitoring:** disparate impact + equalized odds computed pre-promotion
- **Model cards:** structured JSON, machine-readable for regulatory reporting (SR 11-7, EU AI Act)
- **Version control:** every production model is attributable to a specific training run, dataset, and commit

<span class="muted">Designed to satisfy model-risk-management requirements out of the box.</span>

---

## What's Next — Production Hardening Roadmap

| Phase | Scope |
|---|---|
| **1. Foundations** | Docker, docker-compose, externalized config, structured logging |
| **2. Quality Gates** | pytest suite, GitHub Actions CI, pre-commit hooks |
| **3. Serving Hardening** | Auth, rate limiting, Prometheus metrics, OTel tracing |
| **4. Ops & Enforcement** | Airflow/Prefect, Slack/PagerDuty alerts, Grafana dashboards |
| **5. Streaming & Scale** | Kafka ingestion, Redis feature store, Feast PIT correctness |

Each phase is independently shippable.

---

## Why This Platform, Why Now

- **Fraud is an AI-vs-AI arms race.** Static models decay in weeks.
- **Regulators are arriving fast** (EU AI Act, updated SR 11-7). Governance-by-spreadsheet is over.
- **Most teams have a model, not a system.** We have the system.
- **Open, modular, cloud-agnostic.** Deploy on AWS, GCP, Azure, or on-prem.

<div class="lead">

**Our differentiator is not the model — it is the discipline around the model.**

</div>

---

## Summary

<div class="lead">

We have built a **complete, governed, auditable ML lifecycle** for fraud detection — not just a model.

</div>

- Real code, not slideware: ~2,000 LOC of Python across 7 capabilities
- Every architectural choice maps to a known production failure mode
- Ready for hardening into a deployable, multi-tenant SaaS or embedded enterprise platform

---

<!-- _class: lead -->
# Thank You

**Questions & Discussion**

<br>

<span class="muted">
Repository: <code>ltiwariai/enterprise-mlops-platform</code><br>
Branch: <code>claude/understand-project-0cdNv</code>
</span>
