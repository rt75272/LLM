"""
GPT-style decoder-only transformer model.

Architecture:
    token embedding  +  positional embedding
    -> N x TransformerBlock
    -> LayerNorm
    -> Linear (language-model head)
    -> cross-entropy loss

All arithmetic uses plain Python lists via llm.matrix and llm.layers.
No third-party libraries are required.
"""

import math

from .layers import Embedding, Linear, LayerNorm, TransformerBlock
from .matrix import add_mat, add_vec, sum_rows, zeros_mat, zeros_vec, acc_mat, acc_vec, softmax_vec


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def cross_entropy_loss(logits: list, targets: list):
    """
    Compute cross-entropy loss and its gradient w.r.t. logits.

    Parameters
    ----------
    logits  : 2-D list of shape (T, vocab_size)  — raw scores (not softmax'd)
    targets : 1-D list of T integer token ids

    Returns
    -------
    loss       : scalar float
    d_logits   : 2-D list of shape (T, vocab_size)
    """
    T = len(logits)
    V = len(logits[0])

    total_loss = 0.0
    d_logits = [[0.0] * V for _ in range(T)]

    for t in range(T):
        probs = softmax_vec(logits[t])
        target = targets[t]
        total_loss -= math.log(probs[target] + 1e-10)

        # Gradient: probs - one_hot(target), then divide by T
        inv_T = 1.0 / T
        for j in range(V):
            d_logits[t][j] = (probs[j] - (1.0 if j == target else 0.0)) * inv_T

    return total_loss / T, d_logits


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GPT:
    """
    A small GPT-style language model built entirely from scratch.

    Parameters
    ----------
    vocab_size  : number of tokens in the vocabulary
    n_embd      : embedding / model dimension
    n_head      : number of attention heads (must divide n_embd)
    n_layer     : number of transformer blocks
    block_size  : maximum sequence length (context window)
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 64,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 64,
    ):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size

        # Token and positional embeddings
        self.tok_emb = Embedding(vocab_size, n_embd)
        self.pos_emb = Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(n_embd, n_head, block_size)
            for _ in range(n_layer)
        ]

        # Final layer norm + language-model head
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size, bias=False)

        self._cache = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, idx: list) -> list:
        """
        Parameters
        ----------
        idx : 1-D list of T integer token indices  (T <= block_size)

        Returns
        -------
        logits : 2-D list of shape (T, vocab_size)
        """
        T = len(idx)
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        positions = list(range(T))

        # (T, C)
        tok = self.tok_emb.forward(idx)
        pos = self.pos_emb.forward(positions)
        x = add_mat(tok, pos)

        for block in self.blocks:
            x = block.forward(x)

        x = self.ln_f.forward(x)
        logits = self.lm_head.forward(x)   # (T, vocab_size)

        self._cache = (idx, positions, T)
        return logits

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, d_logits: list) -> None:
        """
        Backpropagate gradients from the language-model head all the way
        back to the embeddings, accumulating parameter gradients.

        Parameters
        ----------
        d_logits : 2-D list of shape (T, vocab_size)
        """
        idx, positions, T = self._cache

        # lm_head and final layer norm
        d = self.lm_head.backward(d_logits)  # (T, n_embd)
        d = self.ln_f.backward(d)             # (T, n_embd)

        # Transformer blocks in reverse
        for block in reversed(self.blocks):
            d = block.backward(d)

        # Token embeddings
        self.tok_emb.backward(d)

        # Positional embeddings: gradient is summed over the batch dimension
        # (here batch=1, so pos_grad == d for the single sequence)
        self.pos_emb.backward(d)

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def params(self) -> list:
        """Return list of (parameter, gradient) tuples for the optimiser."""
        p = self.tok_emb.params() + self.pos_emb.params()
        for block in self.blocks:
            p += block.params()
        p += self.ln_f.params() + self.lm_head.params()
        return p

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        self.tok_emb.zero_grad()
        self.pos_emb.zero_grad()
        for block in self.blocks:
            block.zero_grad()
        self.ln_f.zero_grad()
        self.lm_head.zero_grad()

    def num_params(self) -> int:
        """Count total number of scalar parameters."""
        total = 0
        for param, _ in self.params():
            if isinstance(param[0], list):
                total += len(param) * len(param[0])
            else:
                total += len(param)
        return total

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights to a plain-text file (one number per line)."""
        import struct
        with open(path, "wb") as f:
            # Header: vocab_size, n_embd, n_head, n_layer, block_size
            f.write(struct.pack("5i", self.vocab_size, self.n_embd,
                                self.n_head, self.n_layer, self.block_size))
            for param, _ in self.params():
                if isinstance(param[0], list):
                    for row in param:
                        for val in row:
                            f.write(struct.pack("f", val))
                else:
                    for val in param:
                        f.write(struct.pack("f", val))

    @classmethod
    def load(cls, path: str) -> "GPT":
        """Load model weights from a file saved with save()."""
        import struct
        with open(path, "rb") as f:
            vocab_size, n_embd, n_head, n_layer, block_size = struct.unpack(
                "5i", f.read(20)
            )
            model = cls(vocab_size, n_embd, n_head, n_layer, block_size)
            for param, _ in model.params():
                if isinstance(param[0], list):
                    for row in param:
                        for j in range(len(row)):
                            (row[j],) = struct.unpack("f", f.read(4))
                else:
                    for j in range(len(param)):
                        (param[j],) = struct.unpack("f", f.read(4))
        return model
