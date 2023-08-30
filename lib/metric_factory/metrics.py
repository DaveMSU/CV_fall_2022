import typing as tp

import numpy as np
import sklearn.metrics


class _sklearn_metric_handler:
    def __init__(self, metric_type: str, params: dict):
        self._metric_type = metric_type
        self._params = params
        self._metric = getattr(sklearn.metrics, self._metric_type)
        self._worst_value = self._metric([1, 0], [0, 1])
        self._best_value = self._metric([1, 0], [1, 0])
        self._bigger_means_better : bool = self._best_value > self._worst_value
        self._contained_value : tp.Optional[float] = None

    @property
    def worst_value(self) -> float:
        return self._worst_value
    
    @property
    def value(self) -> float:
        return self._contained_value

    def is_it_better_than(self, value: float) -> bool:
        return self._contained_value > value \
            if self._bigger_means_better else self._contained_value < value

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplemented


class _discrete(_sklearn_metric_handler):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.shape == y_pred.shape
        assert y_true.ndim in {1, 2}
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        self._contained_value = self._metric(y_true, y_pred, **self._params)
        return self._contained_value


class _continuous(_sklearn_metric_handler):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.ndim in {1, 2}
        self._contained_value = self._metric(y_true, y_pred, **self._params)
        return self._contained_value


class MetricFactory:
    @staticmethod
    def create_metric(metric_type: str, params: dict):
        assert metric_type in {
            "accuracy_score",
            "f1_score",
            "precision_score",
            "recall_score",
            # "roc_auc_score",  # TODO: support applying map function to
            # "log_loss"  # TODO:  ... prediction before passing in metric.
        }
        if metric_type not in {"roc_auc_score", "log_loss"}:
            return _discrete(metric_type=metric_type, params=params)
        else:
            return _continuous(metric_type=metric_type, params=params)

