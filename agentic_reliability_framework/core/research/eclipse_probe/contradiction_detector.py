"""
Contradiction detector – measures how much the answer contradicts the evidence.
Uses either a small NLI model or cosine similarity (fallback).
"""

import torch
import numpy as np
from typing import Dict, Union, Optional

try:
    from transformers import pipeline
    NLI_AVAILABLE = True
except ImportError:
    NLI_AVAILABLE = False

class ContradictionDetector:
    def __init__(self, method: str = "cosine", nli_model_name: str = "roberta-large-mnli"):
        """
        Args:
            method: "nli" or "cosine"
            nli_model_name: HuggingFace model for NLI (if method="nli")
        """
        self.method = method
        if method == "nli" and NLI_AVAILABLE:
            self.nli_pipeline = pipeline("text-classification", model=nli_model_name)
        elif method == "nli" and not NLI_AVAILABLE:
            print("NLI not available, falling back to cosine similarity.")
            self.method = "cosine"

    def compute_contradiction(self, question: str, answer: str, evidence: str) -> Dict[str, float]:
        """
        Returns a contradiction score (higher means more contradiction).
        """
        if self.method == "nli":
            return self._nli_contradiction(answer, evidence)
        else:
            return self._cosine_contradiction(answer, evidence)

    def _nli_contradiction(self, answer: str, evidence: str) -> Dict[str, float]:
        """Use NLI model: premise = evidence, hypothesis = answer."""
        result = self.nli_pipeline(f"{evidence} </s></s> {answer}")
        label = result[0]['label']
        score = result[0]['score']
        if label.upper() == "CONTRADICTION":
            contradiction_score = score
        elif label.upper() == "NEUTRAL":
            contradiction_score = 0.5 * score
        else:  # ENTAILMENT
            contradiction_score = 1.0 - score
        return {"contradiction_score": contradiction_score, "method": "nli"}

    def _cosine_contradiction(self, answer: str, evidence: str) -> Dict[str, float]:
        """Fallback: use sentence embeddings and cosine distance."""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(answer)
            emb2 = model.encode(evidence)
            cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            contradiction = 1.0 - cos_sim
            return {"contradiction_score": float(contradiction), "method": "cosine"}
        except ImportError:
            # Ultra fallback: word overlap
            words_a = set(answer.lower().split())
            words_e = set(evidence.lower().split())
            if not words_e:
                return {"contradiction_score": 0.5, "method": "fallback"}
            overlap = len(words_a & words_e) / len(words_e)
            contradiction = 1.0 - overlap
            return {"contradiction_score": contradiction, "method": "fallback"}

# For easy use, create a default instance
detector = ContradictionDetector(method="cosine")

def compute_contradiction_score(question: str, answer: str, evidence: str) -> Dict[str, float]:
    """Wrapper function."""
    return detector.compute_contradiction(question, answer, evidence)
