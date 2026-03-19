"""
Entropy estimator for generated answers.
Computes token-level entropy as a measure of model uncertainty.
"""

import torch
import numpy as np
from typing import Dict, Optional, Any

def estimate_answer_entropy(
    model,
    tokenizer,
    question: str,
    evidence: Optional[str] = None,
    max_new_tokens: int = 50,
    temperature: float = 1.0
) -> Dict[str, Any]:
    """
    Generate an answer and compute token-level entropy.

    Args:
        model: HuggingFace model (with .generate and .forward methods)
        tokenizer: HuggingFace tokenizer
        question: input question string
        evidence: optional context string
        max_new_tokens: maximum number of tokens to generate
        temperature: sampling temperature (used only for generation, not for entropy)

    Returns:
        dict containing:
            - entropy: total sequence entropy (sum over tokens)
            - avg_token_entropy: mean entropy per token
            - answer: generated text
            - token_entropies: list of per-token entropies
    """
    # Prepare prompt
    prompt = question
    if evidence:
        prompt = f"Context: {evidence}\n\nQuestion: {question}\n\nAnswer:"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    input_ids = inputs["input_ids"]

    # Generate answer tokens
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    generated_ids = outputs.sequences[0, input_ids.shape[1]:]  # skip prompt
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute log probabilities for each generated token
    scores = outputs.scores  # list of tensors of shape (batch, vocab_size)
    token_log_probs = []
    for i, token_id in enumerate(generated_ids):
        logits = scores[i][0]  # first batch
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_prob = log_probs[token_id].item()
        token_log_probs.append(token_log_prob)

    # Convert log probabilities to probabilities
    token_probs = np.exp(token_log_probs)

    # Compute per-token entropy: -p * log(p) (using natural log)
    token_entropies = [-p * np.log(p) for p in token_probs]
    total_entropy = sum(token_entropies)
    avg_entropy = total_entropy / len(token_entropies) if token_entropies else 0.0

    return {
        "entropy": total_entropy,
        "avg_token_entropy": avg_entropy,
        "answer": generated_text,
        "token_entropies": token_entropies,
        "token_log_probs": token_log_probs,
    }
