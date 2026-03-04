"""
Pure-Python matrix and vector operations.

All tensors are represented as nested Python lists:
  - 1-D vector : [float, ...]                 shape (n,)
  - 2-D matrix : [[float, ...], ...]           shape (rows, cols)

Only the Python standard library (math, random) is used.
"""

import math
import random


# ---------------------------------------------------------------------------
# Random-number helpers
# ---------------------------------------------------------------------------

def _gauss() -> float:
    """Standard-normal sample via the polar form of the Box-Muller transform."""
    while True:
        u = 2.0 * random.random() - 1.0
        v = 2.0 * random.random() - 1.0
        s = u * u + v * v
        if 0.0 < s < 1.0:
            return u * math.sqrt(-2.0 * math.log(s) / s)


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def zeros_vec(n: int):
    return [0.0] * n


def ones_vec(n: int):
    return [1.0] * n


def randn_vec(n: int, std: float = 0.02):
    return [_gauss() * std for _ in range(n)]


def zeros_mat(rows: int, cols: int):
    return [[0.0] * cols for _ in range(rows)]


def ones_mat(rows: int, cols: int):
    return [[1.0] * cols for _ in range(rows)]


def randn_mat(rows: int, cols: int, std: float = 0.02):
    return [[_gauss() * std for _ in range(cols)] for _ in range(rows)]


def copy_vec(v):
    return v[:]


def copy_mat(A):
    return [row[:] for row in A]


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

def matmul(A, B):
    """C = A @ B  —  A is (m, k), B is (k, n), returns (m, n).

    Uses Python's built-in ``sum`` and ``zip`` (both implemented in C) for
    the inner dot product so that this stays competitive without any
    third-party numeric library.
    """
    Bt = transpose(B)          # (n, k) — cache column vectors of B
    return [
        [sum(a * b for a, b in zip(row, col)) for col in Bt]
        for row in A
    ]


def transpose(A):
    """B = A.T  —  A is (m, n), returns (n, m)."""
    m = len(A)
    n = len(A[0])
    return [[A[i][j] for i in range(m)] for j in range(n)]


# ---------------------------------------------------------------------------
# Element-wise operations
# ---------------------------------------------------------------------------

def add_mat(A, B):
    """Element-wise A + B  (both 2-D, same shape)."""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def sub_mat(A, B):
    """Element-wise A - B  (both 2-D, same shape)."""
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def mul_mat(A, B):
    """Element-wise A * B  (both 2-D, same shape)."""
    return [[A[i][j] * B[i][j] for j in range(len(A[0]))]
            for i in range(len(A))]


def scale_mat(A, s: float):
    """Scale every element of 2-D matrix A by scalar s."""
    return [[A[i][j] * s for j in range(len(A[0]))]
            for i in range(len(A))]


def add_bias(A, b):
    """Add 1-D bias vector b to every row of 2-D matrix A."""
    cols = len(A[0])
    return [[A[i][j] + b[j] for j in range(cols)] for i in range(len(A))]


def add_vec(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def sub_vec(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def mul_vec(a, b):
    return [a[i] * b[i] for i in range(len(a))]


def scale_vec(v, s: float):
    return [x * s for x in v]


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def sum_rows(A):
    """Sum every column of A  ->  1-D vector of length cols."""
    cols = len(A[0])
    s = [0.0] * cols
    for row in A:
        for j in range(cols):
            s[j] += row[j]
    return s


def sum_cols(A):
    """Sum every row of A  ->  1-D vector of length rows."""
    return [sum(row) for row in A]


def mean_vec(v):
    return sum(v) / len(v)


def var_vec(v):
    m = mean_vec(v)
    return sum((x - m) ** 2 for x in v) / len(v)


# ---------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------

def gelu(x: float) -> float:
    """Gaussian Error Linear Unit (scalar)."""
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_grad(x: float) -> float:
    """Derivative of GELU at scalar x."""
    phi = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    tanh_phi = math.tanh(phi)
    dphi_dx = math.sqrt(2.0 / math.pi) * (1.0 + 3.0 * 0.044715 * x ** 2)
    return 0.5 * (1.0 + tanh_phi) + 0.5 * x * (1.0 - tanh_phi ** 2) * dphi_dx


def gelu_mat(A):
    """Element-wise GELU on 2-D matrix."""
    return [[gelu(A[i][j]) for j in range(len(A[0]))] for i in range(len(A))]


def gelu_grad_mat(A):
    """Element-wise GELU derivative on 2-D matrix."""
    return [[gelu_grad(A[i][j]) for j in range(len(A[0]))] for i in range(len(A))]


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax_vec(x):
    """Numerically stable softmax of a 1-D list."""
    mx = max(x)
    e = [math.exp(xi - mx) for xi in x]
    s = sum(e)
    return [ei / s for ei in e]


def softmax_mat(A):
    """Row-wise softmax of a 2-D matrix."""
    return [softmax_vec(row) for row in A]


def softmax_backward_mat(d_out, attn):
    """
    Backward pass through row-wise softmax.

    Given the upstream gradient d_out and the forward softmax output attn
    (both 2-D matrices of shape (T, T)), returns the gradient w.r.t. the
    pre-softmax scores.

        d_scores[i] = attn[i] * (d_out[i] - dot(attn[i], d_out[i]))
    """
    T = len(attn)
    d_scores = [[0.0] * T for _ in range(T)]
    for i in range(T):
        a = attn[i]
        d = d_out[i]
        dot = sum(a[j] * d[j] for j in range(T))
        for j in range(T):
            d_scores[i][j] = a[j] * (d[j] - dot)
    return d_scores


# ---------------------------------------------------------------------------
# Layer normalisation
# ---------------------------------------------------------------------------

def layernorm_fwd(x, gamma, beta, eps: float = 1e-5):
    """
    Layer-normalise 1-D vector x with learnable gamma and beta.

    Returns (out, cache) where cache is needed for the backward pass.
    """
    n = len(x)
    mu = sum(x) / n
    var = sum((xi - mu) ** 2 for xi in x) / n
    std_inv = 1.0 / math.sqrt(var + eps)
    xhat = [(xi - mu) * std_inv for xi in x]
    out = [gamma[j] * xhat[j] + beta[j] for j in range(n)]
    return out, (xhat, std_inv)


def layernorm_bwd(d_out, gamma, cache):
    """
    Backward pass through layer norm (1-D).

    Returns (d_x, d_gamma, d_beta).
    """
    xhat, std_inv = cache
    n = len(d_out)

    d_gamma = [d_out[j] * xhat[j] for j in range(n)]
    d_beta = d_out[:]

    d_xhat = [d_out[j] * gamma[j] for j in range(n)]
    s1 = sum(d_xhat)
    s2 = sum(d_xhat[j] * xhat[j] for j in range(n))
    d_x = [std_inv / n * (n * d_xhat[j] - s1 - xhat[j] * s2)
           for j in range(n)]

    return d_x, d_gamma, d_beta


# ---------------------------------------------------------------------------
# Gradient accumulation helpers
# ---------------------------------------------------------------------------

def acc_vec(target, source):
    """target[i] += source[i]  in-place."""
    for i in range(len(target)):
        target[i] += source[i]


def acc_mat(target, source):
    """target[i][j] += source[i][j]  in-place."""
    for i in range(len(target)):
        for j in range(len(target[0])):
            target[i][j] += source[i][j]
