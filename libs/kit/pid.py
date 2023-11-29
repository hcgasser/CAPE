"""PID controller for beta in VAEs"""

import numpy as np


# pylint: disable=invalid-name
def calc_beta(beta, actual, target, I, Kp, Ki, beta_min, beta_max):
    """Inspired by: Huajie Shao et al. “ControlVAE: Controllable Variational Autoencoder”. In:
    arXiv:2004.05988 [cs, stat] (June 20, 2020).
    """

    error = target - actual

    # pylint: disable=invalid-name
    P = Kp / (1 + np.exp(error))

    if beta_min < beta < beta_max:
        I = I - Ki * error

    beta = min(max(P + I + beta_min, beta_min), beta_max)
    return beta, I
