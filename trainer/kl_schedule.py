
import numpy as np


def frange_cycle_linear(n_iter, beta_min=0.0, beta_max=1.0, n_cycle=4, ratio=0.5):
    """ 
    Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing
    https://github.com/haofuml/cyclical_annealing

    Creates cyclical schedule for beta increase
    
    Args:
        n_iter: total iterations
        beta_min: minimum KL weight. Defaults to 0.0.
        beta_max: maximum KL weight. Defaults to 1.0.
        n_cycle: number of cycles. Defaults to 4.
        ratio: percent of iterations spent increasing beta. Defaults to 0.5.
    Returns:
        beta schedule
    """
    L = np.ones(n_iter) * beta_max
    period = n_iter / n_cycle
    step = (beta_max - beta_min) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = beta_min, 0
        while v <= beta_max and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def capacity_increase(n_iter, perc_iter_increase, c_min=0, c_max=25):
    """ 
    Understanding disentangling in Beta-VAE

    Creates monotinic schedule for increasing C 
    Loss = reconstruction + gamma * | KL - C |
    
    Args:
        n_iter: total iterations
        perc_iter_increase: percent of iterations increasing C
        c_min: minimum C
        c_max: maximum C
    Returns:
        C schedule
    """
    increase = (c_max - c_min) / int(n_iter * perc_iter_increase)
    return np.clip(np.arange(0, n_iter) * increase + c_min, c_min, c_max)




def kl_scheduler(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.75, schedule = "linear"):
    """ 
    Allows for two types of scheduling for parameters (beta, C ) in two different
    papers on KL training strategies

    Understanding disentangling in Beta-VAE (C is allowable KL capacity)

    and 

    Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing (beta is KL weight)
    
    Args:
        n_iter: total iterations
        start: minimum param value for beta or C. Defaults to 0.0.
        stop: maximum KL weight for beta or C. Defaults to 1.0.
        n_cycle: number of cycles. Defaults to 4.
        ratio: percent of iterations spent increasing. Defaults to 0.5.
        strategy: cyclical or linear
    Returns:
        param schedule
    """

    if schedule == "cyclical":
        #https://github.com/haofuml/cyclical_annealing 
        flip = stop < start 
        if flip:
            tmp_start = stop
            stop = start
            start = tmp_start   

        L = np.ones(n_iter) * stop
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule

        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        if flip:
            return stop - L
        return L

    increase = (stop - start) / int(n_iter * ratio)
    return np.clip(np.arange(0, n_iter) * increase + start, min(start, stop), max(start, stop))


def start_stop(beta_start, beta_stop, C_start, C_stop, kl_loss):

    if kl_loss == "capacity":
        return C_start, C_stop

    return beta_start, beta_stop
