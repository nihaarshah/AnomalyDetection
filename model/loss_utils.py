# Includes code borrowed from github.com/rtqichen/ffjord/

import math

import torch
import torch.utils.data

MIN_EPSILON = 1e-5
MAX_EPSILON = 1.0 - 1e-5

PI = torch.FloatTensor([math.pi])
if torch.cuda.is_available():
    PI = PI.cuda()

# N(x | mu, var) = 1/sqrt{2pi var} exp[-1/(2 var) (x-mean)(x-mean)]
# log N(x| mu, var) = -log sqrt(2pi) -0.5 log var - 0.5 (x-mean)(x-mean)/var


def log_normal_diag(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_normalized(x, mean, log_var, average=False, reduce=True, dim=None):
    log_norm = -(x - mean) * (x - mean)
    log_norm *= torch.reciprocal(2.0 * log_var.exp())
    log_norm += -0.5 * log_var
    log_norm += -0.5 * torch.log(2.0 * PI)

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm


def log_normal_standard(x, average=False, reduce=True, dim=None):
    log_norm = -0.5 * x * x

    if reduce:
        if average:
            return torch.mean(log_norm, dim)
        else:
            return torch.sum(log_norm, dim)
    else:
        return log_norm
