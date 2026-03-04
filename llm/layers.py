"""
Neural-network layers for the from-scratch transformer.

Every layer exposes:
    forward(x)   -> y          (stores activations needed for backward)
    backward(dy) -> dx         (accumulates gradients into self.grad_*)
    params()     -> [(param, grad), ...]

All tensors are plain Python lists / lists-of-lists.
No third-party packages are used.
"""

import math

from .matrix import (
    zeros_vec, ones_vec, randn_vec,
    zeros_mat, randn_mat,
    copy_vec, copy_mat,
    matmul, transpose,
    add_mat, sub_mat, mul_mat, scale_mat,
    add_bias, add_vec, sub_vec, mul_vec, scale_vec,
    sum_rows,
    gelu_mat, gelu_grad_mat,
    softmax_vec, softmax_mat, softmax_backward_mat,
    layernorm_fwd, layernorm_bwd,
    acc_vec, acc_mat,
)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

class Embedding:
    """
    Lookup table: integer indices -> dense vectors.

    Parameters
    ----------
    num_embeddings : vocabulary size
    embedding_dim  : dimension of each embedding vector
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = randn_mat(num_embeddings, embedding_dim, std=0.02)
        self.grad_weight = zeros_mat(num_embeddings, embedding_dim)
        self._idx = None

    def forward(self, idx: list) -> list:
        """
        idx  : flat list of T integer token / position indices
        returns: 2-D list of shape (T, embedding_dim)
        """
        self._idx = idx
        return [self.weight[i][:] for i in idx]

    def backward(self, d_out: list) -> None:
        """Accumulate gradients; returns None (no gradient for indices)."""
        for t, i in enumerate(self._idx):
            acc_vec(self.grad_weight[i], d_out[t])

    def zero_grad(self):
        for row in self.grad_weight:
            for j in range(len(row)):
                row[j] = 0.0

    def params(self):
        return [(self.weight, self.grad_weight)]


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class Linear:
    """
    Fully-connected layer:  y = x @ W + b

    Parameters
    ----------
    in_features  : input dimension
    out_features : output dimension
    bias         : whether to include a bias term
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        std = 0.02
        self.weight = randn_mat(in_features, out_features, std=std)
        self.grad_weight = zeros_mat(in_features, out_features)
        self.bias = zeros_vec(out_features) if bias else None
        self.grad_bias = zeros_vec(out_features) if bias else None
        self._x = None

    def forward(self, x: list) -> list:
        """x : (T, in_features)  ->  (T, out_features)"""
        self._x = x
        out = matmul(x, self.weight)
        if self.bias is not None:
            out = add_bias(out, self.bias)
        return out

    def backward(self, d_out: list) -> list:
        """
        d_out : (T, out_features)
        returns d_x : (T, in_features)
        """
        # Gradient w.r.t. weight:  x.T @ d_out
        acc_mat(self.grad_weight, matmul(transpose(self._x), d_out))
        # Gradient w.r.t. bias:  sum over T
        if self.bias is not None:
            acc_vec(self.grad_bias, sum_rows(d_out))
        # Gradient w.r.t. input:  d_out @ W.T
        return matmul(d_out, transpose(self.weight))

    def zero_grad(self):
        for row in self.grad_weight:
            for j in range(len(row)):
                row[j] = 0.0
        if self.grad_bias is not None:
            for j in range(len(self.grad_bias)):
                self.grad_bias[j] = 0.0

    def params(self):
        p = [(self.weight, self.grad_weight)]
        if self.bias is not None:
            p.append((self.bias, self.grad_bias))
        return p


# ---------------------------------------------------------------------------
# Layer Normalisation
# ---------------------------------------------------------------------------

class LayerNorm:
    """
    Layer normalisation applied independently to each row (token position).

    y = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, n_embd: int, eps: float = 1e-5):
        self.gamma = ones_vec(n_embd)
        self.beta = zeros_vec(n_embd)
        self.grad_gamma = zeros_vec(n_embd)
        self.grad_beta = zeros_vec(n_embd)
        self.eps = eps
        self._caches = None

    def forward(self, x: list) -> list:
        """x : (T, n_embd)  ->  (T, n_embd)"""
        out_rows = []
        caches = []
        for row in x:
            y, cache = layernorm_fwd(row, self.gamma, self.beta, self.eps)
            out_rows.append(y)
            caches.append(cache)
        self._caches = caches
        return out_rows

    def backward(self, d_out: list) -> list:
        """d_out : (T, n_embd)  ->  d_x : (T, n_embd)"""
        d_x = []
        for t, (d_row, cache) in enumerate(zip(d_out, self._caches)):
            dx_row, dg_row, db_row = layernorm_bwd(d_row, self.gamma, cache)
            d_x.append(dx_row)
            acc_vec(self.grad_gamma, dg_row)
            acc_vec(self.grad_beta, db_row)
        return d_x

    def zero_grad(self):
        for j in range(len(self.grad_gamma)):
            self.grad_gamma[j] = 0.0
            self.grad_beta[j] = 0.0

    def params(self):
        return [(self.gamma, self.grad_gamma), (self.beta, self.grad_beta)]


# ---------------------------------------------------------------------------
# Causal Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class CausalSelfAttention:
    """
    Multi-head causal (masked) self-attention.

    The sequence can attend only to positions <= the current position
    (enforced by a large negative bias on future entries of the score matrix).
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Combined QKV projection  (in: n_embd, out: 3 * n_embd)
        self.c_attn = Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = Linear(n_embd, n_embd)

        # Pre-built causal mask: mask[i][j] = 0 if j<=i else -1e10
        self._mask = [
            [0.0 if j <= i else -1e10 for j in range(block_size)]
            for i in range(block_size)
        ]

        # Cache for backward
        self._cache = None

    def forward(self, x: list) -> list:
        """x : (T, C)  ->  (T, C)"""
        T = len(x)
        C = self.n_embd
        H = self.n_head
        D = self.head_dim

        # --- QKV projection: (T, C) @ (C, 3C) -> (T, 3C)
        qkv = self.c_attn.forward(x)

        # --- Split into Q, K, V each (T, C)
        q = [[qkv[t][j]       for j in range(0,     C)] for t in range(T)]
        k = [[qkv[t][j]       for j in range(C,   2*C)] for t in range(T)]
        v = [[qkv[t][j]       for j in range(2*C, 3*C)] for t in range(T)]

        mask = self._mask

        head_outs = []   # each (T, D)
        head_caches = [] # (q_h, k_h, v_h, attn_h) per head

        for h in range(H):
            lo, hi = h * D, (h + 1) * D

            # Extract head slices (T, D)
            q_h = [[q[t][d] for d in range(lo, hi)] for t in range(T)]
            k_h = [[k[t][d] for d in range(lo, hi)] for t in range(T)]
            v_h = [[v[t][d] for d in range(lo, hi)] for t in range(T)]

            # Scores: (T, T)  =  q_h @ k_h.T * scale + causal_mask
            raw = matmul(q_h, transpose(k_h))
            scores = [
                [raw[i][j] * self.scale + mask[i][j] for j in range(T)]
                for i in range(T)
            ]

            # Attention weights: row-wise softmax  (T, T)
            attn_h = softmax_mat(scores)

            # Weighted values: (T, T) @ (T, D) -> (T, D)
            out_h = matmul(attn_h, v_h)

            head_outs.append(out_h)
            head_caches.append((q_h, k_h, v_h, attn_h))

        # --- Concatenate heads: (T, C)
        concat = [
            [head_outs[h][t][d] for h in range(H) for d in range(D)]
            for t in range(T)
        ]

        # --- Output projection: (T, C) -> (T, C)
        out = self.c_proj.forward(concat)

        self._cache = (q, k, v, head_caches, concat, T, C, H, D)
        return out

    def backward(self, d_out: list) -> list:
        """d_out : (T, C)  ->  d_x : (T, C)"""
        q, k, v, head_caches, concat, T, C, H, D = self._cache

        # --- Backward through output projection
        d_concat = self.c_proj.backward(d_out)   # (T, C)

        # Accumulate head gradients w.r.t. q, k, v
        d_q = [[0.0] * C for _ in range(T)]
        d_k = [[0.0] * C for _ in range(T)]
        d_v = [[0.0] * C for _ in range(T)]

        for h in range(H):
            lo, hi = h * D, (h + 1) * D
            q_h, k_h, v_h, attn_h = head_caches[h]

            # Gradient for this head's output: (T, D)
            d_out_h = [[d_concat[t][lo + d] for d in range(D)] for t in range(T)]

            # d_attn_h = d_out_h @ v_h.T   (T, T)
            d_attn_h = matmul(d_out_h, transpose(v_h))

            # d_v_h = attn_h.T @ d_out_h   (T, D)
            d_v_h = matmul(transpose(attn_h), d_out_h)

            # Backward through softmax  (T, T)
            d_scores = softmax_backward_mat(d_attn_h, attn_h)

            # Apply scale
            d_scores = scale_mat(d_scores, self.scale)

            # d_q_h = d_scores @ k_h   (T, D)
            d_q_h = matmul(d_scores, k_h)

            # d_k_h = d_scores.T @ q_h (T, D)
            d_k_h = matmul(transpose(d_scores), q_h)

            # Scatter head gradients back into full d_q, d_k, d_v
            for t in range(T):
                for d in range(D):
                    d_q[t][lo + d] += d_q_h[t][d]
                    d_k[t][lo + d] += d_k_h[t][d]
                    d_v[t][lo + d] += d_v_h[t][d]

        # --- Reassemble d_qkv: (T, 3C)
        d_qkv = [
            d_q[t] + d_k[t] + d_v[t]
            for t in range(T)
        ]

        # --- Backward through QKV projection
        d_x = self.c_attn.backward(d_qkv)
        return d_x

    def zero_grad(self):
        self.c_attn.zero_grad()
        self.c_proj.zero_grad()

    def params(self):
        return self.c_attn.params() + self.c_proj.params()


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward:
    """
    Position-wise FFN: Linear -> GELU -> Linear

    Inner dimension is 4 * n_embd (standard transformer sizing).
    """

    def __init__(self, n_embd: int):
        self.fc1 = Linear(n_embd, 4 * n_embd)
        self.fc2 = Linear(4 * n_embd, n_embd)
        self._cache = None

    def forward(self, x: list) -> list:
        """x : (T, n_embd)  ->  (T, n_embd)"""
        h = self.fc1.forward(x)       # (T, 4*n_embd)
        h_act = gelu_mat(h)            # (T, 4*n_embd)
        out = self.fc2.forward(h_act)  # (T, n_embd)
        self._cache = (h, h_act)
        return out

    def backward(self, d_out: list) -> list:
        h, h_act = self._cache
        d_h_act = self.fc2.backward(d_out)       # (T, 4*n_embd)
        d_h = mul_mat(d_h_act, gelu_grad_mat(h)) # (T, 4*n_embd)
        return self.fc1.backward(d_h)             # (T, n_embd)

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()

    def params(self):
        return self.fc1.params() + self.fc2.params()


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock:
    """
    One transformer decoder block:

        x1 = x  + Attention(LayerNorm1(x))
        x2 = x1 + FFN(LayerNorm2(x1))
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int):
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd)
        self._cache = None

    def forward(self, x: list) -> list:
        """x : (T, C)  ->  (T, C)"""
        # Attention sub-layer with residual
        ln1_out = self.ln1.forward(x)
        attn_out = self.attn.forward(ln1_out)
        x1 = add_mat(x, attn_out)

        # FFN sub-layer with residual
        ln2_out = self.ln2.forward(x1)
        ffn_out = self.ffn.forward(ln2_out)
        x2 = add_mat(x1, ffn_out)

        self._cache = (x, x1)
        return x2

    def backward(self, d_out: list) -> list:
        """d_out : (T, C)  ->  d_x : (T, C)"""
        x, x1 = self._cache

        # Backward through FFN sub-layer
        d_ffn = self.ffn.backward(d_out)
        d_ln2 = self.ln2.backward(d_ffn)
        d_x1 = add_mat(d_out, d_ln2)   # residual

        # Backward through attention sub-layer
        d_attn = self.attn.backward(d_x1)
        d_ln1 = self.ln1.backward(d_attn)
        d_x = add_mat(d_x1, d_ln1)     # residual

        return d_x

    def zero_grad(self):
        self.ln1.zero_grad()
        self.attn.zero_grad()
        self.ln2.zero_grad()
        self.ffn.zero_grad()

    def params(self):
        return (self.ln1.params() + self.attn.params() +
                self.ln2.params() + self.ffn.params())
