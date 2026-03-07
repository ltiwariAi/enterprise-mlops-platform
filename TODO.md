# TODO — Enterprise MLOps Platform

## Post-Build 1
- [ ] Dockerize all components (serving, training, monitoring, registry)
- [ ] Kubernetes deployment — orchestrate containers, auto-scaling for serving layer
- [ ] Airflow/Prefect DAGs to schedule drift checks and retraining jobs
- [ ] Prometheus + Grafana or build a custom  dash??

## Post-Build 2
- [ ] Governance gates integrated into K8s deployment pipeline
- [ ] Model promotion requires passing all gates before container rollout

## Post-Build 3
- [ ] Kafka streaming for real-time feature ingestion
- [ ] Redis feature store deployed as K8s service
- [ ] Data catalog API as separate microservice

## Data
- [ ] Synthetic transaction dataset from personal transaction data (anonymized)
- [ ] Replace Kaggle dataset to demonstrate richer feature engineering
