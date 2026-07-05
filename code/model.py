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
        return self.h @ self.W2 + self.b2 * tt

    def step(self, xt, t, target):
        pred = self.forward(xt, t)
        tt = t / 200.0
        
        gradOut = 2 * (pred - target) / self.N
        
        gW2 = np.outer(self.h, gradOut)
        gb2 = gradOut * tt
        
        grad_h = gradOut @ self.W2.T
        grad_h[self.h <= 0] = 0 # ReLU derivative
        
        gW1 = np.outer(xt, grad_h)
        gb1 = grad_h * tt
        
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        return pred
