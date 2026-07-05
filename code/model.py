import numpy as np

# stand-in for the transformer noise predictor from the proposal.
# using a linear model here since this is just the phase-1 prototype,
# swap this out for the real thing in phase 2
class NoisePredictor:
    def __init__(self, N, hidden=64, lr=0.01):
        self.N = N
        self.W1 = np.random.randn(N, hidden) * 0.1
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, N) * 0.1
        self.b2 = np.zeros(N)
        self.lr = lr

    def forward(self, xt, t):
        tt = t / 200.0
        self.h = np.maximum(0, xt @ self.W1 + self.b1 * tt) # ReLU
        self.logits = self.h @ self.W2 + self.b2 * tt
        
        # Softmax ensures valid probability distribution
        exp_L = np.exp(self.logits - np.max(self.logits))
        self.out = exp_L / np.sum(exp_L)
        return self.out

    def step(self, xt, t, gradOut):
        # Backprop through Softmax
        grad_logits = self.out * (gradOut - np.sum(self.out * gradOut))
        
        tt = t / 200.0
        
        gW2 = np.outer(self.h, grad_logits)
        gb2 = grad_logits * tt
        
        grad_h = grad_logits @ self.W2.T
        grad_h[self.h <= 0] = 0 # ReLU derivative
        
        gW1 = np.outer(xt, grad_h)
        gb1 = grad_h * tt
        
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        return self.out
