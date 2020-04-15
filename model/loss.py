import torch.nn.functional as F

from .loss_utils import *


def mse_loss(output, target):
    return F.mse_loss(output, target)


def kl_divergence(z, z_mu, z_var):
    """KL divergence for non-flow VAE

    Args:
        z: posterior estimate sample
        z_mu: posterior estimate mean
        z_var: posterior estimate var
    """

    log_q_z = log_normal_diag(z, mean=z_mu, log_var=z_var.log(), dim=1)
    log_p_z = log_normal_standard(z, dim=1)

    return torch.sum(log_q_z - log_p_z)


def vae_loss(output, target, z, z_mu, z_var):

    beta = 1
    # import pickle
    # pickle.dump((z, z_mu, z_var), open("test.pkl", "wb"))
    # raise ValueError
    return mse_loss(output, target) + beta * kl_divergence(z, z_mu, z_var)
