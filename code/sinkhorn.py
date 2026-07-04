import numpy as np

# quick simplex projector. calling it "sinkhorn" because that's what the
# proposal calls the Proj_delta step, even though this is just clip+renorm,
# not full entropic OT sinkhorn (that part lives in barycenter.py instead)
def sinkhorn_projection(x, n_iter=5):
    x = np.array(x, dtype=float)
    for i in range(n_iter):
        x = np.clip(x, 1e-8, None)
        x = x / x.sum()
    return x


def test():
    v = np.array([0.5, -0.2, 0.9, 0.0])
    out = sinkhorn_projection(v)
    print(out, out.sum())


if __name__ == "__main__":
    test()
