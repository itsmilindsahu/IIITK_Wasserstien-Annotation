import numpy as np

# stand-in for the transformer noise predictor from the proposal.
# using a linear model here since this is just the phase-1 prototype,
# swap this out for the real thing in phase 2
class NoisePredictor:
    def __init__(self, N, lr=0.05):
        self.N = N
        self.W = np.random.randn(N, N) * 0.01
        self.b = np.zeros(N)
        self.lr = lr

    def forward(self, xt, t):
        tt = t / 200.0
        return self.W @ xt + self.b * tt

    def step(self, xt, t, target):
        pred = self.forward(xt, t)
        gradOut = 2 * (pred - target) / self.N
        gW = np.outer(gradOut, xt)
        gb = gradOut * (t / 200.0)
        self.W -= self.lr * gW
        self.b -= self.lr * gb
        return pred
