import numpy as np
from scipy.stats import norm


def max_uncertainty_reduction_query(regressor, X, n_instances=1):
    """Maximum uncertainty reduction acquisition function. Refines an already well-performing model.

    Parameters
    ----------
    regressor : estimator object
        Regressor.
    X : np.array
        Pool of examples.

    Returns
    -------
    index : int
        Query index.
    X_i : np.array
        Query instance.
    """
    # Predict on available points
    _, std = regressor.predict(X, return_std=True)

    # Take largest N standard devs
    query_idx = np.argsort(std)[-n_instances:]

    return query_idx, X[query_idx]


def exploitation_query(regressor, X, n_instances=1):
    """Returns the instance within `X` with highest predicted value.

    Parameters
    ----------
    regressor : estimator object
        Regressor.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances.

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
    query_idx = np.argsort(preds)[-n_instances:]

    return query_idx, X[query_idx]


def mutual_information_query(regressor, X, n_instances=1):
    """Selects instances with the highest mutual information, i.e., where predictions are both uncertain and informative.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor with `predict(X, return_std=True)`.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances.

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
    query_idx = np.argsort(mutual_info)[-n_instances:]

    return query_idx, X[query_idx]


def expected_improvement_query(regressor, X, best_y, n_instances=1, xi=0.01):
    """
    Expected Improvement (EI) acquisition function. Balances exploration and exploitation.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor.
    X : np.array
        Pool of examples.
    best_y : float
        Best observed value so far.
    n_instances : int
        Number of instances to select.
    xi : float
        Exploration-exploitation tradeoff parameter.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    mean, std = regressor.predict(X, return_std=True)
    std = std.clip(min=1e-9)  # Avoid division by zero

    improvement = mean - best_y - xi
    Z = improvement / std
    EI = improvement * norm.cdf(Z) + std * norm.pdf(Z)

    query_idx = np.argsort(EI)[-n_instances:]

    return query_idx, X[query_idx]


def upper_confidence_bound_query(regressor, X, n_instances=1, beta=2.0):
    """
    Upper Confidence Bound (UCB) acquisition function. Ensures exploration while still considering high predictions.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances to select.
    beta : float
        Tradeoff parameter (higher = more exploration).

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    mean, std = regressor.predict(X, return_std=True)

    ucb = mean + beta * std  # Exploration-exploitation balance

    query_idx = np.argsort(ucb)[-n_instances:]

    return query_idx, X[query_idx]


def thompson_sampling_query(regressor, X, n_instances=1):
    """
    Thompson Sampling acquisition function. Injects stochasticity into the selection process,
    making exploration more adaptive.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances to select.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    mean, std = regressor.predict(X, return_std=True)

    sampled_values = np.random.normal(mean, std)  # Sample from GP posterior

    query_idx = np.argsort(sampled_values)[-n_instances:]

    return query_idx, X[query_idx]


def knowledge_gradient_query(regressor, X, n_instances=1):
    """
    Knowledge Gradient (KG) acquisition function. Explicitly models how knowledge gained now will benefit future
    queries.

    Parameters
    ----------
    regressor : estimator object
        Gaussian Process regressor.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances to select.

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

    query_idx = np.argsort(kg)[-n_instances:]

    return query_idx, X[query_idx]


def random_query(regressor, X, n_instances=1):
    """
    Random acquisition function. Randomly selects points from the pool. Useful as null model.

    Parameters
    ----------
    regressor : estimator object
        Ignored.
    X : np.array
        Pool of examples.
    n_instances : int
        Number of instances to select.

    Returns
    -------
    query_idx : int
        Query index.
    X_i : np.array
        Query instance.
    """
    query_idx = np.random.choice(X.shape[0], n_instances, replace=False)

    return query_idx, X[query_idx]
