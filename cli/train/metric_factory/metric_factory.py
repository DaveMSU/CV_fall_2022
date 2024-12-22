import typing as tp

import numpy as np
import sklearn.metrics

from . import additional_metrics, transforms
from ..learning_config import OneMetricConfig
from lib import wrap_in_logger


class ScikitLearnMetricHandler:
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def __init__(
            self,
            metric_type: str,
            params: tp.Dict[str, tp.Any],
            target_transform: transforms.BaseTransform,
            prediction_transform: transforms.BaseTransform
    ):
        # sel._metric_type = metric_type
        self._params = params
        self._metric: tp.Callable[
                [np.ndarray, np.ndarray],  # shapes: (bs,)
                np.Union[np.float64, float]
        ]
        try:
            self._metric = getattr(additional_metrics, metric_type)
        except AttributeError:
            self._metric = getattr(sklearn.metrics, metric_type)
        self._metric = wrap_in_logger(level="debug")(self._metric)
        self._worst_value = float(self._metric([1, 0], [0, 1]))
        self._best_value = float(self._metric([1, 0], [1, 0]))
        self._bigger_means_better: bool = self._best_value > self._worst_value
        self._target_transform: transforms.BaseTransform = target_transform
        self._prediction_transform: transforms.BaseTransform = prediction_transform  # noqa
    
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def __call__(
            self,
            y_true: np.ndarray,  # shape: (bs,) / (bs, class_amount)
            y_pred: np.ndarray  # shape: (bs,) / (bs, class_amount)
    ) -> float:
        y_true = self._target_transform(y_true)
        y_pred = self._prediction_transform(y_pred)
        return float(self._metric(y_true, y_pred, **self._params))

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
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def best_value(self) -> float:
        return self._worst_value

    @property
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def worst_value(self) -> float:
        return self._worst_value

    @classmethod
    def from_config(
            cls,
            config: OneMetricConfig
    ) -> 'ScikitLearnMetricHandler':
        return cls(
            config.function,
            config.params,
            target_transform=getattr(
                transforms,
                config.target_transform.type
            )(
                **config.target_transform.params
            ),
            prediction_transform=getattr(
                transforms,
                config.prediction_transform.type
            )(
                **config.prediction_transform.params
            )
        )

    @wrap_in_logger(level="trace", ignore_args=(0,))
    def is_first_better_than_second(
            self,
            /,
            first_value: float,
            second_value: float
    ) -> bool:
        return first_value > second_value \
            if self._bigger_means_better else first_value < second_value
