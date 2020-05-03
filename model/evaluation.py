import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def top_n_percent_anomaly(recon_error, true, dataset="kdd"):

    if dataset == "kdd":
        top_n_perc = 0.2
    elif dataset == "thyroid":
        top_n_perc = 0.025
    elif dataset == "musk":
        top_n_perc = 0.032
    else:
        top_n_perc = 0.15

    n_obs = len(recon_error)
    top_n = int(n_obs * top_n_perc)
    top_recon_error_idx = np.argsort(recon_error)[-top_n:]

    predicted = np.zeros(n_obs)
    predicted[top_recon_error_idx] = 1

    return {
        "n": n_obs,
        "accuracy": round(accuracy_score(true, predicted), 5),
        "f1": round(f1_score(true, predicted), 5),
        "recall": round(recall_score(true, predicted), 5),
        "precision": round(precision_score(true, predicted), 5),
    }


# def reconstruction_probability()
