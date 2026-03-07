"""
Governance Gate — Production Approval Workflow
Enterprise MLOps Platform

No model reaches production without passing ALL gates.
"""

import json
from datetime import datetime


class GovernanceGate:
    """Automated governance checks before production deployment."""

    def __init__(self, min_auc_pr: float = 0.75, min_recall: float = 0.70):
        self.min_auc_pr = min_auc_pr
        self.min_recall = min_recall
        self.audit_log = []

    def _log(self, action: str, result: str, details: dict = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "result": result,
            "details": details or {},
        }
        self.audit_log.append(entry)
        return entry

    def check_performance(self, metrics: dict) -> dict:
        """Gate 1: Model meets minimum performance thresholds."""
        auc_pr = metrics.get("auc_pr", 0)
        recall = metrics.get("recall", 0)

        passed = auc_pr >= self.min_auc_pr and recall >= self.min_recall

        result = {
            "gate": "performance",
            "passed": passed,
            "auc_pr": auc_pr,
            "auc_pr_threshold": self.min_auc_pr,
            "recall": recall,
            "recall_threshold": self.min_recall,
        }

        self._log("performance_check", "passed" if passed else "failed", result)
        return result

    def check_bias(self, bias_results: list) -> dict:
        """Gate 2: Model passes bias checks across all groups."""
        all_passed = all(r["overall_passed"] for r in bias_results)
        failed_groups = [r["group_name"] for r in bias_results if not r["overall_passed"]]

        result = {
            "gate": "bias",
            "passed": all_passed,
            "groups_checked": len(bias_results),
            "groups_failed": failed_groups,
        }

        self._log("bias_check", "passed" if all_passed else "failed", result)
        return result

    def check_explainability(self, global_importance, min_features: int = 5) -> dict:
        """Gate 3: Model predictions are explainable with sufficient feature diversity."""
        top_features = len(global_importance[global_importance['mean_abs_shap'] > 0.01])

        passed = top_features >= min_features

        result = {
            "gate": "explainability",
            "passed": passed,
            "meaningful_features": top_features,
            "min_required": min_features,
            "top_feature": global_importance.iloc[0]['feature'],
            "top_feature_dominance": round(
                global_importance.iloc[0]['mean_abs_shap'] / global_importance['mean_abs_shap'].sum(), 4
            ),
        }

        self._log("explainability_check", "passed" if passed else "failed", result)
        return result

    def check_model_card(self, card: dict) -> dict:
        """Gate 4: Model card exists with all required fields."""
        required_sections = ["model_details", "intended_use", "training_data",
                           "performance", "ethical_considerations", "limitations"]

        missing = [s for s in required_sections if s not in card]
        passed = len(missing) == 0

        result = {
            "gate": "model_card",
            "passed": passed,
            "missing_sections": missing,
        }

        self._log("model_card_check", "passed" if passed else "failed", result)
        return result

    def run_all_gates(self, metrics: dict, bias_results: list,
                      global_importance, model_card: dict) -> dict:
        """Run all governance gates and return final decision."""

        print(f"\nGOVERNANCE GATE — PRODUCTION APPROVAL")
        print(f"{'='*60}")
        print(f"Timestamp: {datetime.now().isoformat()}")

        g1 = self.check_performance(metrics)
        print(f"\n  Gate 1 — Performance:     {'PASSED' if g1['passed'] else 'FAILED'}")
        print(f"    AUC-PR: {g1['auc_pr']:.4f} (min: {g1['auc_pr_threshold']})")
        print(f"    Recall: {g1['recall']:.4f} (min: {g1['recall_threshold']})")

        g2 = self.check_bias(bias_results)
        print(f"\n  Gate 2 — Bias:            {'PASSED' if g2['passed'] else 'FAILED'}")
        print(f"    Groups checked: {g2['groups_checked']}")
        if g2['groups_failed']:
            print(f"    Failed groups: {', '.join(g2['groups_failed'])}")

        g3 = self.check_explainability(global_importance)
        print(f"\n  Gate 3 — Explainability:  {'PASSED' if g3['passed'] else 'FAILED'}")
        print(f"    Meaningful features: {g3['meaningful_features']} (min: {g3['min_required']})")
        print(f"    Top feature dominance: {g3['top_feature_dominance']*100:.1f}%")

        g4 = self.check_model_card(model_card)
        print(f"\n  Gate 4 — Model Card:      {'PASSED' if g4['passed'] else 'FAILED'}")
        if g4['missing_sections']:
            print(f"    Missing: {', '.join(g4['missing_sections'])}")

        all_passed = g1['passed'] and g2['passed'] and g3['passed'] and g4['passed']

        print(f"\n{'='*60}")
        if all_passed:
            print(f"  DECISION: APPROVED FOR PRODUCTION")
        else:
            failed_gates = []
            if not g1['passed']: failed_gates.append("performance")
            if not g2['passed']: failed_gates.append("bias")
            if not g3['passed']: failed_gates.append("explainability")
            if not g4['passed']: failed_gates.append("model_card")
            print(f"  DECISION: BLOCKED — failed gates: {', '.join(failed_gates)}")
        print(f"{'='*60}")

        decision = {
            "approved": all_passed,
            "timestamp": datetime.now().isoformat(),
            "gates": {
                "performance": g1,
                "bias": g2,
                "explainability": g3,
                "model_card": g4,
            },
            "audit_log": self.audit_log,
        }

        return decision

    def save_audit_log(self, path: str = "governance/audit/audit_log.json"):
        with open(path, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        print(f"Audit log saved: {path}")
