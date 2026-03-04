"""
Adam optimiser — implemented from scratch with no third-party libraries.

Reference: Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014).
"""

import math


class Adam:
    """
    Adam optimiser.

    Parameters
    ----------
    params       : list of (param, grad) tuples as returned by model.params()
    lr           : learning rate (α)
    beta1        : exponential decay rate for the first moment (default 0.9)
    beta2        : exponential decay rate for the second moment (default 0.999)
    eps          : numerical stability constant (default 1e-8)
    weight_decay : L2 regularisation coefficient (default 0.0)
    """

    def __init__(
        self,
        params: list,
        lr: float = 3e-4,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # time-step counter

        # First and second moment estimates — same structure as params
        self.m = []
        self.v = []
        for param, _ in params:
            if isinstance(param[0], list):
                self.m.append([[0.0] * len(param[0]) for _ in param])
                self.v.append([[0.0] * len(param[0]) for _ in param])
            else:
                self.m.append([0.0] * len(param))
                self.v.append([0.0] * len(param))

    def step(self) -> None:
        """Apply one Adam update step to all parameters."""
        self.t += 1
        b1t = self.beta1 ** self.t
        b2t = self.beta2 ** self.t
        # Bias-corrected learning rate
        alpha = self.lr * math.sqrt(1.0 - b2t) / (1.0 - b1t)

        for i, (param, grad) in enumerate(self.params):
            if isinstance(param[0], list):
                # 2-D parameter (matrix)
                rows = len(param)
                cols = len(param[0])
                for r in range(rows):
                    for c in range(cols):
                        g = grad[r][c]
                        if self.weight_decay != 0.0:
                            g = g + self.weight_decay * param[r][c]
                        self.m[i][r][c] = (self.beta1 * self.m[i][r][c]
                                           + (1.0 - self.beta1) * g)
                        self.v[i][r][c] = (self.beta2 * self.v[i][r][c]
                                           + (1.0 - self.beta2) * g * g)
                        param[r][c] -= (alpha * self.m[i][r][c]
                                        / (math.sqrt(self.v[i][r][c]) + self.eps))
            else:
                # 1-D parameter (vector)
                n = len(param)
                for j in range(n):
                    g = grad[j]
                    if self.weight_decay != 0.0:
                        g = g + self.weight_decay * param[j]
                    self.m[i][j] = (self.beta1 * self.m[i][j]
                                    + (1.0 - self.beta1) * g)
                    self.v[i][j] = (self.beta2 * self.v[i][j]
                                    + (1.0 - self.beta2) * g * g)
                    param[j] -= (alpha * self.m[i][j]
                                 / (math.sqrt(self.v[i][j]) + self.eps))
