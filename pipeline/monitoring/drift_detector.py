"""
Drift Detection Service
Enterprise MLOps Platform
"""

import numpy as np
import pandas as pd
from datetime import datetime


class DriftDetector:
    """
    Monitors incoming data for distribution drift against training baseline.
    Uses PSI as primary signal. Supports critical feature watchlist.
    """

    def __init__(self, reference_data: pd.DataFrame, psi_threshold: float = 0.2,
                 critical_features: list = None):
        self.reference = reference_data
        self.psi_threshold = psi_threshold
        self.critical_features = critical_features or []
        self.alerts = []

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1
        )

        expected_counts = np.histogram(expected, bins=breakpoints)[0] + 1
        actual_counts = np.histogram(actual, bins=breakpoints)[0] + 1

        expected_pct = expected_counts / expected_counts.sum()
        actual_pct = actual_counts / actual_counts.sum()

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def check_drift(self, current_data: pd.DataFrame, exclude_features: list = None) -> dict:
        exclude = exclude_features or []

        results = {
            "timestamp": datetime.now().isoformat(),
            "samples_checked": len(current_data),
            "features_drifted": [],
            "critical_features_drifted": [],
            "overall_status": "healthy",
            "details": {}
        }

        for col in self.reference.columns:
            if col not in current_data.columns or col in exclude:
                continue

            ref_values = self.reference[col].dropna().values
            cur_values = current_data[col].dropna().values

            if len(cur_values) < 10:
                continue

            psi = self.calculate_psi(ref_values, cur_values)

            ref_mean = ref_values.mean()
            cur_mean = cur_values.mean()
            mean_shift = abs(cur_mean - ref_mean) / (ref_values.std() + 1e-6)

            is_drifted = psi > self.psi_threshold

            results["details"][col] = {
                "psi": round(psi, 4),
                "mean_shift_std": round(mean_shift, 4),
                "ref_mean": round(ref_mean, 4),
                "cur_mean": round(cur_mean, 4),
                "drifted": is_drifted,
                "is_critical": col in self.critical_features
            }

            if is_drifted:
                results["features_drifted"].append(col)
                if col in self.critical_features:
                    results["critical_features_drifted"].append(col)

        # Determine status
        drift_count = len(results["features_drifted"])
        total_features = len(results["details"])
        drift_pct = drift_count / total_features if total_features > 0 else 0

        if results["critical_features_drifted"]:
            results["overall_status"] = "critical"
        elif drift_pct > 0.3:
            results["overall_status"] = "critical"
        elif drift_pct > 0.1:
            results["overall_status"] = "warning"

        if results["overall_status"] != "healthy":
            self.alerts.append({
                "timestamp": results["timestamp"],
                "status": results["overall_status"],
                "drifted_features": drift_count,
                "critical_drifted": results["critical_features_drifted"],
                "total_features": total_features,
                "recommendation": "retrain" if results["overall_status"] == "critical" else "monitor"
            })

        return results

    def get_alerts(self) -> list:
        return self.alerts

    def should_retrain(self) -> bool:
        if not self.alerts:
            return False
        return self.alerts[-1]["status"] == "critical"
