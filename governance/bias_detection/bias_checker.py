"""
Bias Detection and Fairness Analysis
Enterprise MLOps Platform

Checks if the model's predictions are fair across demographic groups.
A model CANNOT reach production if bias thresholds are exceeded.
"""

import numpy as np
import pandas as pd


class BiasChecker:
    """
    Measures model fairness across protected groups.
    Uses disparate impact ratio and equalized odds.
    """

    def __init__(self, disparate_impact_threshold: float = 0.8,
                 equalized_odds_threshold: float = 0.1):
        self.di_threshold = disparate_impact_threshold
        self.eo_threshold = equalized_odds_threshold

    def check_bias(self, y_true: np.ndarray, y_pred: np.ndarray,
                   y_prob: np.ndarray, group_labels: np.ndarray,
                   group_name: str = "group") -> dict:
        """
        Run bias analysis across groups.

        Args:
            y_true: Actual labels (0/1)
            y_pred: Predicted labels (0/1)
            y_prob: Predicted probabilities
            group_labels: Group membership for each sample
            group_name: Name of the grouping variable

        Returns:
            dict with bias metrics and pass/fail status
        """
        groups = np.unique(group_labels)

        group_metrics = {}
        for g in groups:
            mask = group_labels == g
            n = mask.sum()
            if n < 10:
                continue

            g_true = y_true[mask]
            g_pred = y_pred[mask]
            g_prob = y_prob[mask]

            tp = ((g_pred == 1) & (g_true == 1)).sum()
            fp = ((g_pred == 1) & (g_true == 0)).sum()
            fn = ((g_pred == 0) & (g_true == 1)).sum()
            tn = ((g_pred == 0) & (g_true == 0)).sum()

            flag_rate = g_pred.mean()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            avg_score = g_prob.mean()

            group_metrics[str(g)] = {
                "count": int(n),
                "fraud_count": int(g_true.sum()),
                "flag_rate": round(flag_rate, 6),
                "false_positive_rate": round(fpr, 6),
                "false_negative_rate": round(fnr, 6),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "avg_fraud_score": round(avg_score, 6),
            }

        # Disparate Impact Ratio
        flag_rates = {g: m["flag_rate"] for g, m in group_metrics.items() if m["flag_rate"] > 0}
        if flag_rates:
            max_rate = max(flag_rates.values())
            min_rate = min(flag_rates.values())
            di_ratio = min_rate / max_rate if max_rate > 0 else 1.0
        else:
            di_ratio = 1.0

        # Equalized Odds — max difference in FPR and FNR across groups
        fprs = [m["false_positive_rate"] for m in group_metrics.values()]
        fnrs = [m["false_negative_rate"] for m in group_metrics.values()]
        eo_fpr_diff = max(fprs) - min(fprs) if fprs else 0
        eo_fnr_diff = max(fnrs) - min(fnrs) if fnrs else 0
        eo_max_diff = max(eo_fpr_diff, eo_fnr_diff)

        # Pass/fail
        di_passed = di_ratio >= self.di_threshold
        eo_passed = eo_max_diff <= self.eo_threshold
        overall_passed = di_passed and eo_passed

        result = {
            "group_name": group_name,
            "num_groups": len(group_metrics),
            "group_metrics": group_metrics,
            "disparate_impact_ratio": round(di_ratio, 4),
            "disparate_impact_threshold": self.di_threshold,
            "disparate_impact_passed": di_passed,
            "equalized_odds_fpr_diff": round(eo_fpr_diff, 4),
            "equalized_odds_fnr_diff": round(eo_fnr_diff, 4),
            "equalized_odds_max_diff": round(eo_max_diff, 4),
            "equalized_odds_threshold": self.eo_threshold,
            "equalized_odds_passed": eo_passed,
            "overall_passed": overall_passed,
        }

        return result

    def print_report(self, result: dict):
        """Print bias analysis report."""
        print(f"\nBIAS ANALYSIS — {result['group_name']}")
        print(f"{'='*60}")

        print(f"\n{'Group':<15} {'Count':>8} {'Fraud':>6} {'Flag%':>8} {'FPR':>8} {'FNR':>8} {'Recall':>8}")
        print(f"{'-'*63}")

        for g, m in result['group_metrics'].items():
            print(f"{g:<15} {m['count']:>8,} {m['fraud_count']:>6} {m['flag_rate']*100:>7.3f}% {m['false_positive_rate']*100:>7.3f}% {m['false_negative_rate']*100:>7.3f}% {m['recall']*100:>7.1f}%")

        print(f"\nDisparate Impact Ratio: {result['disparate_impact_ratio']:.4f} (threshold: >= {result['disparate_impact_threshold']})")
        print(f"  {'PASSED' if result['disparate_impact_passed'] else 'FAILED'}")

        print(f"\nEqualized Odds Max Diff: {result['equalized_odds_max_diff']:.4f} (threshold: <= {result['equalized_odds_threshold']})")
        print(f"  FPR diff: {result['equalized_odds_fpr_diff']:.4f}")
        print(f"  FNR diff: {result['equalized_odds_fnr_diff']:.4f}")
        print(f"  {'PASSED' if result['equalized_odds_passed'] else 'FAILED'}")

        print(f"\nOVERALL: {'*** PASSED ***' if result['overall_passed'] else '*** FAILED — BLOCKED FROM PRODUCTION ***'}")
