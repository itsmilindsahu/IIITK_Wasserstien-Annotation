import numpy as np

from forward import dirichlet_noise, forward_step_mine, forward_step_baseline


def generate_sample(net, N, T=200, mode="mine", rng=None):
    """
    Actually run the reverse diffusion process end-to-end to produce a
    generated sample from the trained noise predictor `net`, rather than
    returning x_star (the barycenter/mean target used only to build the
    training signal).

    The network in model.py is trained with an x0-parameterization: at
    each step it is asked to predict the clean target directly from a
    noisy xt (see the `eps = x_star` comment in train.py), not the noise
    itself. So the reverse process here is the standard "predict x0, then
    re-noise to a smaller t" (DDIM-style) sampler:

        x_T           ~ noise
        for t = T .. 1:
            x0_hat     = net.forward(x_t, t)
            x_{t-1}    = forward_step(x0_hat, t-1)   if t > 1
                       = x0_hat                       if t == 1

    Using the *predicted* x0 (not the true x_star, which is unavailable at
    generation time) to re-noise at each step means later steps can correct
    earlier ones, and by t=1 the chain has actually been driven by the
    learned network the whole way -- this is what should be compared to
    held-out annotators, not x_star's distance to the annotators that were
    averaged into it in the first place.

    mode: "mine" uses the Dirichlet forward process + simplex projection;
          "baseline" uses the Gaussian forward process + clip/renormalize.
    """
    if rng is None:
        rng = np.random.default_rng()

    if mode == "mine":
        xt = dirichlet_noise(N, alpha=1.0)
    else:
        # start from a valid point on the simplex for the baseline too,
        # for a fair comparison (uniform + small noise, then renormalize)
        xt = np.ones(N) / N

    for t in range(T, 0, -1):
        x0_hat = net.forward(xt, t)
        if t > 1:
            if mode == "mine":
                xt = forward_step_mine(x0_hat, t - 1, T)
            else:
                xt = forward_step_baseline(x0_hat, t - 1, T)
        else:
            xt = x0_hat

    return xt
