import typing as tp
from collections.abc import Sequence

import numpy as np
import sklearn.metrics


def root_mean_squared_error(*args, **kwargs) -> np.float64:
    return sklearn.metrics.mean_squared_error(*args, **kwargs) ** 0.5


def entropy(
        y_true: Sequence[tp.Union[int, float]],
        y_pred: Sequence[tp.Union[int, float]]
) -> float:
    y_pred = np.asarray(y_pred)
    return -(y_pred * np.log(y_pred)).sum() / y_pred.shape[0]


def log_loss_dev_by_10(*args, **kwargs) -> float:
    return sklearn.metrics.log_loss(*args, **kwargs) / 10.


def log_loss_diff(
        y_true: Sequence[tp.Union[int, float]],
        y_pred: Sequence[tp.Union[int, float]]
) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    best_const: np.array
    if y_true.ndim == 1:
        best_const = y_true.mean() * np.ones(y_true.size)
    elif y_true.ndim == 2:
        best_const = np.eye(y_true.shape[1])[
            np.ones(y_true.shape[0], dtype=int) * y_true.sum(axis=0).argmax()
        ]
    else:
        raise ValueError(
            "Wrong ndim has been found: {y_true.ndim} (must be 1 or 2)."
        )
    ll_best_const: float = sklearn.metrics.log_loss(y_true, best_const)
    return sklearn.metrics.log_loss(y_true, y_pred) - ll_best_const


# TODO: understand this metric before start to use it.
# def log_loss_p(y_true: np.array, y_pred: np.array) -> float:
#     y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
#     best_const: np.array
#     if y_true.ndim == 1:
#         norm_coef: float = y_true.mean()
#         best_const = norm_coef * np.ones(y_true.size)
#     elif y_true.ndim == 2:
#         norm_coef: float = y_true.mean(axis=0).max()
#         best_const = np.eye(y_true.shape[1])[
#             np.ones(y_true.shape[0], dtype=int) * y_true.sum(axis=0).argmax()
#         ]
#     else:
#         raise ValueError(
#             "Wrong ndim has been found: {y_true.ndim} (must be 1 or 2)."
#         )
#     ll_best_const: float = sklearn.metrics.log_loss(y_true, best_const)
#     ll: float = sklearn.metrics.log_loss(y_true, y_pred)
#     return (ll_best_const - ll) / norm_coef

