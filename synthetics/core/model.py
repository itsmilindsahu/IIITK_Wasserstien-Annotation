"""
core/model.py

Two-layer ReLU MLP noise predictor.

Maps (x_t, t) -> x̂₀ (clean target prediction, x0-parameterisation).
The final softmax ensures the output is always a valid probability distribution
on the simplex.  Backprop is implemented analytically (no autograd dependency).
"""
import numpy as np


class NoisePredictor:
    """
    Two-layer ReLU network:  x_t -> Linear(N, H) -> ReLU -> Linear(H, N) -> Softmax.

    Parameters
    ----------
    N      : number of frames / simplex dimension
    hidden : hidden layer width (default 64)
    lr     : learning rate (default 0.01)
    rng    : np.random.Generator for reproducible weight init.
             Pass the *same* seeded Generator for two configs being compared
             so that weight-init luck cannot bias the comparison.
    """

    def __init__(self, N, hidden=64, lr=0.01, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.N = N
        self.W1 = rng.standard_normal((N, hidden)) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, N)) * 0.1
        self.b2 = np.zeros(N)
        self.lr = lr

        # Cache for backward pass
        self._h = None
        self._out = None

    def forward(self, xt, t):
        """Forward pass; caches activations for backward."""
        tt = t / 200.0
        self._h = np.maximum(0.0, xt @ self.W1 + self.b1 * tt)
        logits = self._h @ self.W2 + self.b2 * tt
        exp_L = np.exp(logits - logits.max())
        self._out = exp_L / exp_L.sum()
        return self._out

    def step(self, xt, t, grad_out):
        """SGD update given upstream gradient w.r.t. the softmax output."""
        out = self._out
        h = self._h
        tt = t / 200.0

        # Softmax Jacobian: ∂softmax / ∂logit
        grad_logits = out * (grad_out - np.dot(out, grad_out))

        gW2 = np.outer(h, grad_logits)
        gb2 = grad_logits * tt

        grad_h = grad_logits @ self.W2.T
        grad_h[h <= 0] = 0.0  # ReLU derivative

        gW1 = np.outer(xt, grad_h)
        gb1 = grad_h * tt

        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
