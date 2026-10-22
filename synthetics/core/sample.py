"""
core/sample.py

Reverse diffusion sampler (DDIM-style x0-parameterisation):

    x_T ~ noise
    for t = T … 1:
        x̂₀   = net.forward(x_t, t)
        x_{t-1} = forward_step(x̂₀, t-1)   if t > 1
                = x̂₀                        if t == 1

The network is trained to predict x* directly (x0-parameterisation), so
we re-noise the predicted clean target at each reverse step rather than
predicting the added noise.
"""
import numpy as np
from core.forward import dirichlet_noise, forward_step_mine, forward_step_baseline


def generate_sample(net, N, T=200, mode="mine", rng=None):
    """
    Run full reverse diffusion from noise and return the final generated sample.

    Parameters
    ----------
    net  : trained NoisePredictor
    N    : simplex dimension (number of frames)
    T    : total diffusion steps
    mode : "mine" (Dirichlet) | "baseline" (Gaussian)
    rng  : np.random.Generator

    Returns
    -------
    x : (N,) sample on the probability simplex
    """
    rng = rng or np.random.default_rng()

    if mode == "mine":
        xt = dirichlet_noise(N, alpha=1.0, rng=rng)
    else:
        xt = np.ones(N) / N  # start from uniform

    for t in range(T, 0, -1):
        x0_hat = net.forward(xt, t)
        if t > 1:
            if mode == "mine":
                xt = forward_step_mine(x0_hat, t - 1, T, rng=rng)
            else:
                xt = forward_step_baseline(x0_hat, t - 1, T, rng=rng)
        else:
            xt = x0_hat

    return xt
