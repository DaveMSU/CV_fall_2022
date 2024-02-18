import typing as tp

import numpy as np
import sklearn.metrics

from . import additional_metrics


class sklearn_metric_handler:
    def __init__(self, metric_type: str, params: dict):
        self._metric_type = metric_type
        self._params = params
        try:
            self._metric = getattr(additional_metrics, metric_type)
        except AttributeError:
            self._metric = getattr(sklearn.metrics, metric_type)
        self._worst_value = self._metric([1, 0], [0, 1])
        self._best_value = self._metric([1, 0], [1, 0])
        self._bigger_means_better : bool = self._best_value > self._worst_value

    @property
    def worst_value(self) -> float:
        return self._worst_value
    
    def is_first_better_than_second(
            self,
            first_value: float,
            second_value: float
    ) -> bool:
        return first_value > second_value \
            if self._bigger_means_better else first_value < second_value

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplemented


class discrete(sklearn_metric_handler):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.shape == y_pred.shape
        assert y_true.ndim in {1, 2}
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        return self._metric(y_true, y_pred, **self._params)


class continuous(sklearn_metric_handler):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.ndim in {1, 2}
        return self._metric(y_true, y_pred, **self._params)


class MetricFactory:
    _names_of_continuous_metrics = [
        "log_loss",
        "log_loss_dev_by_10",
        "log_loss_diff",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_error",
        "roc_auc_score",
        "root_mean_squared_error"
    ]
    _names_of_discrete_metrics = [
        "accuracy_score",
        "f1_score",
        "precision_score",
        "recall_score"
    ]

    @classmethod
    def create_metric(cls, metric_type: str, params: dict) -> sklearn_metric_handler:
        if metric_type in cls._names_of_continuous_metrics:
            return continuous(metric_type=metric_type, params=params)
        elif metric_type in cls._names_of_discrete_metrics:
            return discrete(metric_type=metric_type, params=params)
        else:
            raise ValueError(f"Unknown metric type: `{metric_type}`.")
