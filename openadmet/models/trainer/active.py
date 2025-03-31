from typing import Any

import numpy as np
from modal.acquisition import max_EI, max_PI, max_UCB
from modal.disagreement import max_std_sampling

from openadmet.models.trainer.trainer_base import TrainerBase, trainers


def exploitation_query(regressor, X, n_samples=1):
    """Custom query strategy for Gaussian processes. Returns the instance within `X` with highest predicted value.

    Parameters
    ----------
    regressor : estimator object
        Regressor.
    X : np.array
        Pool of examples.
    n_samples : int
        Number of samples.

    Returns
    -------
    index : int
        Query index.
    X_i : np.array
        Query instance.
    """
    # Predict on available points
    preds, _ = regressor.predict(X, return_std=True)

    # Take largest N predictions
    query_idx = np.argsort(preds)[-n_samples:]

    return query_idx, X[query_idx]


def mutual_information_query(regressor, X, n_samples=1):
    """Selects instances with the highest mutual information, i.e., where predictions are both uncertain and informative.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    n_samples : int
        Number of samples.

    Returns
    -------
    index : int
        Query index.
    X_i : np.array
        Query instance.
    """
    # Predict mean and standard deviation
    mean, std = regressor.predict(X, return_std=True)

    # Compute mutual information estimate: log variance-based heuristic
    mutual_info = 0.5 * np.log(1 + (std**2))

    # Select points with highest mutual information
    query_idx = np.argsort(mutual_info)[-n_samples:]

    return query_idx, X[query_idx]


def knowledge_gradient_query(regressor, X, n_samples=1):
    """
    Knowledge Gradient (KG) acquisition function. Explicitly models how knowledge gained now will benefit future
    queries.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor.
    X : np.array
        Pool of examples.
    n_samples : int
        Number of samples to select.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    mean, std = regressor.predict(X, return_std=True)

    # KG approximation: prioritize points where std is large but also high mean
    kg = mean + 0.5 * std

    query_idx = np.argsort(kg)[-n_samples:]

    return query_idx, X[query_idx]


def thompson_sampling_query(regressor, X, n_samples=1):
    """
    Thompson Sampling acquisition function. Injects stochasticity into the selection process,
    making exploration more adaptive.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor.
    X : np.array
        Pool of examples.
    n_samples : int
        Number of samples to select.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    mean, std = regressor.predict(X, return_std=True)

    sampled_values = np.random.normal(mean, std)  # Sample from GP posterior

    query_idx = np.argsort(sampled_values)[-n_samples:]

    return query_idx, X[query_idx]


def random_query(regressor, X, n_samples=1):
    """
    Random acquisition function. Randomly selects points from the pool. Useful as null model.

    Parameters
    ----------
    regressor : estimator object
        Ignored.
    X : np.array
        Pool of examples.
    n_samples : int
        Number of samples to select.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    query_idx = np.random.choice(X.shape[0], n_samples, replace=False)

    return query_idx, X[query_idx]


_QUERY_STRATEGIES = {
    "max-uncertainty-reduction": max_std_sampling,
    "exploitation": exploitation_query,  # Equivalent in modAL?
    "mutual-information": mutual_information_query,  # Equivalent in modAL?
    "max-expected-improvement": max_EI,
    "max-probability-improvement": max_PI,
    "upper-confidence-bound": max_UCB,
    "thompson-sampling": thompson_sampling_query,  # Equivalent in modAL?
    "knoweldge-gradient": knowledge_gradient_query,  # Equivalent in modAL?
    "random": random_query,
}


@trainers.register("ActiveLearningTrainer")
class ActiveLearningTrainer(TrainerBase):
    """
    Basic trainer for active learning models
    """

    query_strategy: str = "random"
    n_initial: int = 0
    n_samples_per_iter: int = 0
    n_iters: int = 0
    rebag: bool = False
    bootstrap: bool = False

    def train(self, X: Any, y: Any):
        model = self.model.model
        model.fit(X, y)
        self.model.model = model
        return self.model
