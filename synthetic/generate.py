import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.preprocessing import MinMaxScaler


def sample_mu(rng, k, d):
    """Sample means of clusters"""
    return np.random.uniform(-rng, rng, (k, d))


def sample_cov(k, d, alpha, beta):
    """Sample covariance of clusters with method in 
    Yibo et. al"""
    cov = []
    for _ in range(k):
        # Initial c_j
        C = np.random.normal(0, 1, (d, d))
        # Magnitude of entries in cov matrix
        s = np.random.uniform(alpha, alpha + beta)
        # Orthoganilze C
        C = np.linalg.qr(C)[0] / s
        cov.append(C.T @ C)

    return cov


def sample_cov_eig(k, d, alpha, beta):
    """Sample covariance of clusters with
    eigenvalue method"""
    cov = []
    for _ in range(k):
        # Initialize c_j
        C = np.random.normal(0, 1, (d, d))
        # Orthoganalize C
        C = np.linalg.qr(C)[0]
        # Generate eigen values diagonal matrix
        eig = np.diag(np.random.uniform(alpha, alpha + beta, d))
        # Create cov matrix
        cov.append(C.T @ eig @ C)

    return cov


def generate_dataset(n, d, k, rng, alpha, beta):
    """Generate a dataset"""
    mu = sample_mu(rng, k, d)
    cov = sample_cov(k, d, alpha, beta)

    data = []
    labels = []
    for i in range(k):
        data.append(np.random.multivariate_normal(mean=mu[i, :], cov=cov[i], size=n))
        labels.append(np.repeat(i, n))

    return np.vstack(data), np.hstack(labels)


def generate_outlier_dataset(n, anom_perc, d, rng, alpha, beta):
    """Generate a dataset"""
    mu = sample_mu(rng, 2, d)
    cov = sample_cov(2, d, alpha, beta)

    normal_size = int((1 - anom_perc) * n)
    anom_size = normal_size - tes_normal_size
    test_data = np.vstack(
        [
            np.random.multivariate_normal(mean=mu[0, :], cov=cov[0], size=test_normal_size),
            np.random.multivariate_normal(mean=mu[1, :], cov=cov[1], size=test_anom_size),
        ]
    )
    test_labels = np.hstack([np.repeat(0, test_normal_size), np.repeat(1, test_anom_size)])

    return train_normal_data, test_data, train_normal_labels, test_labels


if __name__ == "__main__":
    circle_X, circle_y = make_circles(10000, noise=0.01)
    circle_X = circle_X[circle_y == 1, :]
    circle_y = circle_y[circle_y == 1]

    pickle.dump((circle_X, circle_y), open("../data/circle_train_normal.pickle", "wb"))
    test_grid = np.array(
        np.meshgrid(
            np.linspace(circle_X[:, 0].min(), circle_X[:, 0].max(), 100),
            np.linspace(circle_X[:, 1].min(), circle_X[:, 1].max(), 100),
        )
    ).T.reshape(-1, 2)
    pickle.dump((test_grid, test_grid), open("../data/circle_test_normal.pickle", "wb"))

    scaler = MinMaxScaler()
    circle_X = scaler.fit_transform(circle_X)
    test_grid = np.array(np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))).T.reshape(
        -1, 2
    )
    pickle.dump((circle_X, circle_y), open("../data/circle_train_binary.pickle", "wb"))
    pickle.dump((test_grid, test_grid), open("../data/circle_test_binary.pickle", "wb"))
