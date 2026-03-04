"""
Autoregressive text generation.

Uses the trained GPT model to produce new tokens one at a time, sampling
from the predicted probability distribution at each step.
"""

import math
import random


def generate(
    model,
    start_tokens: list,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
) -> list:
    """
    Generate *max_new_tokens* new tokens autoregressively.

    Parameters
    ----------
    model          : trained GPT instance
    start_tokens   : list of integer token ids used as the initial prompt
    max_new_tokens : number of tokens to append
    temperature    : softmax temperature — higher = more random (default 1.0)
    top_k          : if set, restrict sampling to the top-k logits

    Returns
    -------
    tokens : list of all token ids (prompt + generated)
    """
    tokens = list(start_tokens)

    for _ in range(max_new_tokens):
        # Crop context to the model's block_size
        context = tokens[-model.block_size:]

        # Forward pass (no gradient needed during inference)
        logits = model.forward(context)

        # Take logits at the last position: shape (vocab_size,)
        last_logits = logits[-1]

        # Apply temperature scaling
        if temperature != 1.0:
            last_logits = [v / temperature for v in last_logits]

        # Optional top-k filtering: zero out all but the top-k entries
        if top_k is not None and top_k < len(last_logits):
            # Find the k-th largest value
            sorted_vals = sorted(last_logits, reverse=True)
            threshold = sorted_vals[top_k - 1]
            last_logits = [
                v if v >= threshold else -1e10
                for v in last_logits
            ]

        # Softmax to get probabilities
        max_l = max(last_logits)
        exp_l = [math.exp(v - max_l) for v in last_logits]
        total = sum(exp_l)
        probs = [e / total for e in exp_l]

        # Sample from the distribution
        r = random.random()
        cumulative = 0.0
        next_token = len(probs) - 1  # fallback
        for idx, p in enumerate(probs):
            cumulative += p
            if r < cumulative:
                next_token = idx
                break

        tokens.append(next_token)

    return tokens
