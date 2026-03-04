"""
Training utilities.

Provides get_batch() for random mini-batch sampling and train() for the
main training loop.  Everything is pure Python — no third-party packages.
"""

import random
import math

from .model import GPT, cross_entropy_loss
from .optimizer import Adam


def get_batch(data: list, block_size: int):
    """
    Sample a random (input, target) pair from *data*.

    Parameters
    ----------
    data       : flat list of integer token ids (the full encoded corpus)
    block_size : context window length

    Returns
    -------
    x : list of block_size integer ids  (input tokens)
    y : list of block_size integer ids  (target tokens — shifted by one)
    """
    max_start = len(data) - block_size - 1
    if max_start <= 0:
        raise ValueError(
            f"Dataset ({len(data)} tokens) is too small for block_size={block_size}."
        )
    i = random.randint(0, max_start)
    x = data[i: i + block_size]
    y = data[i + 1: i + block_size + 1]
    return x, y


def train(
    model: GPT,
    data: list,
    n_steps: int = 1000,
    block_size: int = None,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    eval_interval: int = 100,
    verbose: bool = True,
) -> list:
    """
    Train *model* on *data* for *n_steps* gradient-descent steps.

    One step = one randomly sampled sequence of length *block_size*.

    Parameters
    ----------
    model         : GPT instance
    data          : flat list of integer token ids
    n_steps       : total number of optimisation steps
    block_size    : context length (defaults to model.block_size)
    lr            : Adam learning rate
    weight_decay  : L2 regularisation
    eval_interval : print average loss every this many steps (0 = silent)
    verbose       : whether to print progress

    Returns
    -------
    losses : list of per-step loss values
    """
    if block_size is None:
        block_size = model.block_size

    optimizer = Adam(model.params(), lr=lr, weight_decay=weight_decay)

    losses = []
    window = []  # recent losses for the running average

    for step in range(1, n_steps + 1):
        x, y = get_batch(data, block_size)

        # Forward
        logits = model.forward(x)
        loss, d_logits = cross_entropy_loss(logits, y)

        # Backward
        model.zero_grad()
        model.backward(d_logits)

        # Parameter update
        optimizer.step()

        losses.append(loss)
        window.append(loss)

        if verbose and eval_interval > 0 and step % eval_interval == 0 and window:
            avg = sum(window) / len(window)
            window = []
            print(f"  step {step:>6}/{n_steps}  loss {avg:.4f}")

    return losses
