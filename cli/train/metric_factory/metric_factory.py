import abc
import typing as tp

import numpy as np
import sklearn.metrics

from . import additional_metrics
from ..learning_config import OneMetricConfig


class BaseScikitLearnMetricHandler(abc.ABC):
    def __init__(self, metric_type: str, params: tp.Dict[str, tp.Any]):
        # sel._metric_type = metric_type
        self._params = params
        try:
            self._metric = getattr(additional_metrics, metric_type)
        except AttributeError:
            self._metric = getattr(sklearn.metrics, metric_type)
        self._worst_value: float = self._metric([1, 0], [0, 1])
        self._best_value: float = self._metric([1, 0], [1, 0])
        self._bigger_means_better: bool = self._best_value > self._worst_value
    
    @abc.abstractmethod
    def __call__(
            self,
            y_true: np.ndarray,  # shape: (bs,) / (bs, class_amount)
            y_pred: np.ndarray  # shape: (bs,) / (bs, class_amount)
    ) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
                # f"metric_type={self._metric_type}, "
                f"best_value={self._best_value}, "
                f"worst_value={self._worst_value}, "
                f"_bigger_means_better={self._bigger_means_better}"
                f"_metric={self._metric}, "
            ")"
        )

    # @property
    # def metric_type(self) -> str:
    #     return self._metric_type

    @property
    def best_value(self) -> float:
        return self._worst_value

    @property
    def worst_value(self) -> float:
        return self._worst_value

    def is_first_better_than_second(
            self,
            /,
            first_value: float,
            second_value: float
    ) -> bool:
        return first_value > second_value \
            if self._bigger_means_better else first_value < second_value


class _DiscreteScikitLearnMetricHandler(BaseScikitLearnMetricHandler):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        assert y_true.shape == y_pred.shape
        assert y_true.ndim in {1, 2}
        if y_true.ndim == 2:
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
        return self._metric(y_true, y_pred, **self._params)


class _ContinuousScikitLearnMetricHandler(BaseScikitLearnMetricHandler):
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
    def create_metric(
            cls,
            config: OneMetricConfig
    ) -> BaseScikitLearnMetricHandler:
        if config.function in cls._names_of_continuous_metrics:
            _MetricHandler = _ContinuousScikitLearnMetricHandler
        elif config.function in cls._names_of_discrete_metrics:
            _MetricHandler = _DiscreteScikitLearnMetricHandler
        else:
            raise ValueError(f"Unknown metric: `{config.function}`")
        return _MetricHandler(config.function, config.params)
