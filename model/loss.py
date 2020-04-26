import numpy as np
import torch.nn.functional as F

from .loss_utils import *


def mse_loss(output, target):
    return F.mse_loss(output, target, reduce=False).mean(axis=1)


def bce_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target, reduce=False).mean(axis=1)


def huber_loss(output, target):
    return F.smooth_l1_loss(output, target, reduce=False).mean(axis=1)


def kl_divergence(z_mu, z_var, z_0, z_k, ldj):
    """KL divergence for non-flow VAE

    Args:
        z_mu: posterior estimate mean
        z_var: posterior estimate var
        z_0: first stochastic latent variable
        z_k: last stochastic latent variable
        ldj: log det jacobian
    """

    log_p_z = log_normal_standard(z_k, dim=1)
    log_q_z = log_normal_diag(z_0, mean=z_mu, log_var=z_var.log(), dim=1)

    return log_q_z - log_p_z - ldj


def vae_loss(
    output,
    target,
    z_mu,
    z_var,
    z_0,
    z_k,
    ldj,
    kl_weight_param=None,
    gamma=None,
    recon_loss="mse",
    kl_loss="capacity",
    pointwise=False,
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
        z_mu: latent mean
        z_var: latent var
        z_0: first stochastic latent variable
        z_k: last stochastic latent variable
        ldj: log det jacobian,
        kl_weight_param: beta weight or KL capacity. Defaults to None.
        gamma: gamma weight for KL capacity. Defaults to None.
        recon_loss: reconstruction loss. Defaults to "mse".
        kl_loss: capcity or weight. Defaults to "capacity".
        pointwise: return pointwise loss
    Returns:
        vae loss
    """

    if recon_loss == "mse":
        recon = mse_loss(output, target)
    elif recon_loss == "huber":
        recon = huber_loss(output, target)
    elif recon_loss == "bce":
        recon = bce_loss(output, target)

    kld = kl_divergence(z_mu, z_var, z_0, z_k, ldj)

    if kl_loss == "capacity":
        C = kl_weight_param
        kl_term = gamma * (kld - C).abs()

    elif kl_loss == "weight":
        beta = kl_weight_param
        kl_term = beta * kld

    if not pointwise:
        recon = torch.mean(recon)
        kl_term = torch.mean(kl_term)
        kld = torch.mean(kld)

    return recon + kl_term, recon, kld, kl_weight_param
