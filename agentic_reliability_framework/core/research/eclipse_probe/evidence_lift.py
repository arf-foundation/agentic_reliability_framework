"""
Evidence lift module – measures how much evidence changes the likelihood of the answer.
Lift = log P(answer | question + evidence) - log P(answer | question)
"""

import torch
from typing import Dict, Optional

def compute_evidence_lift(
    model,
    tokenizer,
    question: str,
    answer: str,
    evidence: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute log-likelihood lift from evidence.

    Args:
        model: HuggingFace model (must be able to compute logits)
        tokenizer: HuggingFace tokenizer
        question: input question
        answer: the answer string (generated or ground truth)
        evidence: optional evidence string

    Returns:
        dict with:
            - lift: ΔL (positive if evidence helped)
            - log_prob_with_evidence: log P(answer | question + evidence)
            - log_prob_without_evidence: log P(answer | question)
    """
    def log_likelihood(text, context):
        """Compute log likelihood of text given context."""
        full_text = f"{context} {text}"
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"]

        context_ids = tokenizer(context, return_tensors="pt", truncation=True).input_ids[0]
        answer_ids = tokenizer(text, return_tensors="pt", truncation=True).input_ids[0]
        start = len(context_ids)
        end = start + len(answer_ids)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[0]  # shape (seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1)

        total_log_prob = 0.0
        for i, token_id in enumerate(answer_ids):
            pos = start + i
            if pos < len(log_probs):
                total_log_prob += log_probs[pos, token_id].item()
        return total_log_prob

    log_prob_no_evidence = log_likelihood(answer, question)
    if evidence:
        context_with_evidence = f"Context: {evidence}\nQuestion: {question}\nAnswer:"
        log_prob_with_evidence = log_likelihood(answer, context_with_evidence)
    else:
        log_prob_with_evidence = log_prob_no_evidence

    lift = log_prob_with_evidence - log_prob_no_evidence

    return {
        "lift": lift,
        "log_prob_with_evidence": log_prob_with_evidence,
        "log_prob_without_evidence": log_prob_no_evidence,
    }
