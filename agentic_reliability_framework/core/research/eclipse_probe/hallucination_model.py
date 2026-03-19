"""
Hallucination risk model – combines entropy, evidence lift, and contradiction.
"""

from typing import Dict, Any, Optional

class HallucinationRisk:
    """
    Computes a hallucination risk score as a weighted combination of:
    - entropy (model uncertainty)
    - evidence lift (how much evidence increases answer likelihood)
    - contradiction (how much the answer contradicts evidence)
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Args:
            weights: dict with keys 'entropy', 'lift', 'contradiction'.
                    Defaults: entropy=0.5, lift=-0.3, contradiction=0.2
        """
        self.weights = weights or {
            "entropy": 0.5,
            "lift": -0.3,
            "contradiction": 0.2
        }

    def compute_risk(self, entropy: float, evidence_lift: float, contradiction_score: float) -> Dict[str, Any]:
        """
        Compute hallucination risk.

        Args:
            entropy: total or average entropy from entropy_estimator
            evidence_lift: lift value from evidence_lift
            contradiction_score: score from contradiction_detector (0-1)

        Returns:
            dict with:
                - risk_score: weighted sum (higher = more likely hallucination)
                - signals: input signals
                - weights: weights used
        """
        risk = (self.weights["entropy"] * entropy +
                self.weights["lift"] * evidence_lift +
                self.weights["contradiction"] * contradiction_score)

        return {
            "risk_score": risk,
            "signals": {
                "entropy": entropy,
                "evidence_lift": evidence_lift,
                "contradiction": contradiction_score
            },
            "weights": self.weights
        }
