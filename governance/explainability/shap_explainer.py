"""
SHAP Explainability Module
Enterprise MLOps Platform

Every prediction must be explainable.
"""

import shap
import numpy as np
import pandas as pd


class FraudExplainer:
    """Generates SHAP explanations for fraud predictions."""

    def __init__(self, model, feature_names: list):
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(model)

    def explain_prediction(self, features: np.ndarray) -> dict:
        """Explain a single prediction."""
        if features.ndim == 1:
            features = features.reshape(1, -1)

        shap_values = self.explainer.shap_values(features)

        # For binary classification, shap_values is for positive class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        base_value = self.explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1]

        # Sort by absolute impact
        feature_impacts = []
        for i, (name, value) in enumerate(zip(self.feature_names, sv)):
            feature_impacts.append({
                "feature": name,
                "shap_value": round(float(value), 6),
                "feature_value": round(float(features[0][i]), 4),
                "direction": "fraud" if value > 0 else "legitimate",
                "abs_impact": abs(float(value)),
            })

        feature_impacts.sort(key=lambda x: x["abs_impact"], reverse=True)

        return {
            "base_value": round(float(base_value), 6),
            "prediction_shift": round(float(sum(sv)), 6),
            "top_factors": feature_impacts[:10],
            "all_factors": feature_impacts,
        }

    def explain_batch(self, X: np.ndarray, top_n: int = 5) -> list:
        """Explain multiple predictions."""
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        explanations = []
        for i in range(len(X)):
            impacts = sorted(
                zip(self.feature_names, sv[i]),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n]

            explanations.append({
                "top_factors": [
                    {"feature": name, "shap_value": round(float(val), 6),
                     "direction": "fraud" if val > 0 else "legitimate"}
                    for name, val in impacts
                ]
            })

        return explanations

    def global_importance(self, X: np.ndarray) -> pd.DataFrame:
        """Global feature importance across all predictions."""
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)

        return importance
