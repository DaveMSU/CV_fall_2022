import abc
import dataclasses
import typing as tp
from collections import OrderedDict

import numpy as np

from .metric_factory import BaseScikitLearnMetricHandler, MetricFactory
from ..learning_config import ManyMetricsConfig


@dataclasses.dataclass(frozen=True)
class MetricValueContainer:
    main_metric_name: str
    all: tp.Dict[str, float]  # OrderedDict()


@dataclasses.dataclass(frozen=True)
class MetricHandlerContainer:
    main_metric_name: str
    all: tp.Dict[str, BaseScikitLearnMetricHandler]  # OrderedDict()

    def __post_init__(self):
        _main_has_been_seen = False
        assert isinstance(self.all, dict), type(self.all)
        assert type(self.all) is OrderedDict
        for metric_name, metric_handler in self.all.items():
            if not isinstance(metric_handler, BaseScikitLearnMetricHandler):
                raise TypeError(
                    "Must be at least sub-instance of"
                    " BaseScikitLearnMetricHandler class, but"
                    f" got `{type(metric_handler)}`"
                )
            if metric_name == self.main_metric_name:
                _main_has_been_seen |= True
        else:
            if _main_has_been_seen is False:
                raise AttributeError(
                    f"Main metric `{self.main_metric_name}` doesn't preserve"
                    f" among the 'all': `{getattr(self, 'all').keys()}`"
                )

    def __call__(
            self,
            y_true: np.ndarray,  # shape: (bs,) / (bs, class_amount)
            y_pred: np.ndarray  # shape: (bs,) / (bs, class_amount)
    ) -> MetricValueContainer:
        assert y_true.ndim in [1, 2] and y_pred.ndim in [1, 2]
        assert y_true.shape[0] == y_pred.shape[0]
        assert (y_true.ndim == y_pred.ndim == 1) or (y_true.shape[1] == y_pred.shape[1])  # noqa
        return MetricValueContainer(
            self.main_metric_name,
            OrderedDict(
                [
                    (name, metric_handler(y_true, y_pred))
                    for name, metric_handler in self.all.items()
                ]
            )
        )

    @classmethod
    def from_config(cls, cnfg: ManyMetricsConfig) -> 'MetricHandlerContainer':
        return cls(
            cnfg.main,
            OrderedDict(
                [
                    (  # TODO: don't like that name isn't being used
                        one_metric_cnfg.name,
                        MetricFactory.create_metric(one_metric_cnfg)
                    ) for one_metric_cnfg in cnfg.all
                ]
            )
        )
