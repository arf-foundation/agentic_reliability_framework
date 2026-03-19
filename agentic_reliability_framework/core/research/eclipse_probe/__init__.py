"""
Eclipse Probe – hallucination risk detection using entropy, evidence lift, and contradiction.
"""

from .entropy_estimator import estimate_answer_entropy
from .evidence_lift import compute_evidence_lift
from .contradiction_detector import compute_contradiction_score
from .hallucination_model import HallucinationRisk

__all__ = [
    "estimate_answer_entropy",
    "compute_evidence_lift",
    "compute_contradiction_score",
    "HallucinationRisk",
]
