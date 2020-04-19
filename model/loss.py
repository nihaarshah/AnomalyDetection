import numpy as np
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from .loss_utils import *


def mse_loss(output, target):
    return F.mse_loss(output, target)


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


def huber_loss(output, target):
    return F.smooth_l1_loss(output, target)


def kl_divergence(z, z_mu, z_var):
    """KL divergence for non-flow VAE

    Args:
        z: posterior estimate sample
        z_mu: posterior estimate mean
        z_var: posterior estimate var
    """

    log_q_z = log_normal_diag(z, mean=z_mu, log_var=z_var.log(), dim=1)
    log_p_z = log_normal_standard(z, dim=1)

    return torch.mean(log_q_z - log_p_z)


def vae_loss(
    output,
    target,
    z,
    z_mu,
    z_var,
    kl_weight_param=None,
    gamma=None,
    recon_loss="mse",
    kl_loss="capacity",
):
    """
    Implements two types of loss:

    - Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing 
    - Beta-Vae Learning Basic Visual Concepts with Constrained Variational Framework
    Both papers use a constrained optimization where KL < beta (weight)
    
    Loss = reconstruction + beta * KL

    - Understanding disentangling in Beta-VAE (capacity)

    Loss = reconstruction + gamma * | KL - C |
    
    Args:
        output: VAE output
        target: target reconstruction
        z: latent sample
        z_mu: latent mean
        z_var: latent var
        kl_weight_param: beta weight or KL capacity. Defaults to None.
        gamma: gamma weight for KL capacity. Defaults to None.
        recon_loss: reconstruction loss. Defaults to "mse".
        kl_loss: capcity or weight. Defaults to "capacity".
    Returns:
        vae loss
    """

    if recon_loss == "mse":
        recon = mse_loss(output, target)
    elif recon_loss == "huber":
        recon = huber_loss(output, target)
    elif recon_loss == "bce":
        recon = bce_loss(output, target)

    kld = kl_divergence(z, z_mu, z_var)

    if kl_loss == "capacity":
        C = kl_weight_param
        kl_term = gamma * (kld - C).abs()

    elif kl_loss == "weight":
        beta = kl_weight_param
        kl_term = beta * kld

    return recon + kl_term, recon, kld, kl_weight_param


def top_n_percent_anomaly(recon_error, true, dataset="kdd"):

    top_n_perc = 0.2 if dataset == "kdd" else 0.15
    n_obs = len(recon_error)
    top_n = int(n_obs * top_n_perc)
    top_recon_error_idx = np.argsort(recon_error)[-top_n:]

    predicted = np.zeros(n_obs)
    predicted[top_recon_error_idx] = 1

    return {
        "accuracy": round(accuracy_score(true, predicted), 5),
        "f1": round(f1_score(true, predicted), 5),
        "recall": round(recall_score(true, predicted), 5),
        "precision": round(precision_score(true, predicted), 5),
    }
