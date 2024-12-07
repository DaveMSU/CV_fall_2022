import abc
import dataclasses
import logging
import typing as tp

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from .learning_config import UpdationLevel
from .metric_factory import MetricHandlerContainer, MetricValueContainer
from .training_context import TrainingContext
from lib import (
    LearningMode,
    wrap_in_logger,
)

T = tp.TypeVar('T', float, np.ndarray)


# TODO: want to improve this class and move it to the lib dir
#  it could be used for gradient accumulation algo, eponential running means
#  like in batch norm realisation so I will be able see several
#  running stats not for several sample amounts, it could be helpfull
#  for example when epoch is huge like in LM tasks.
class _RunningMeanHandler(tp.Generic[T]):
    def __init__(self):
        self._value: tp.Optional[T] = None
        self._counter: int = 0

    def __repr__(self) -> str:
        t_type: tp.Type[T] = tp.get_args(self.__orig_class__)[0]
        if self._value is None:
            value_repr = repr(None)
        elif t_type is float:
            value_repr = repr(self._value)
        elif t_type is np.ndarray:
            value_repr = repr((self._value ** 2).sum() ** 0.5) + "(l2)"
        else:
            assert False, t_type
        return (
            f"{self.__class__.__name__}=("
                f"_value={value_repr}, "
                f"_counter={self._counter}"
            ")"
        )

    @wrap_in_logger(level="trace")
    def add(self, value: tp.Optional[T], n: int = 1) -> None:
        self._counter += n
        if value is None:
            return
        elif type(value) is not (t := tp.get_args(self.__orig_class__)[0]):
            raise TypeError(
                f"The value `{value}` has to have"
                f" type `{t}`, got `{type(value)}`"
            )
        elif self._value is None:
            self._value = value
        else:
            self._value += (value - self._value) * (n / self._counter)

    @wrap_in_logger(level="trace", ignore_args=(0,))
    def get_value(self) -> tp.Optional[T]:  # shape: (int,)
        return self._value

    @wrap_in_logger(level="trace", ignore_args=(0,))
    def is_empty(self) -> bool:
        assert self._value is None
        return self._counter == 0

    @wrap_in_logger(level="debug", ignore_args=(0,))  # TODO: may be debug -> trace
    def flush(self) -> tp.Optional[T]:
        value = self._value
        self.__init__()
        assert self.is_empty()
        return value


_FloatRunningMeanHandler = _RunningMeanHandler[float]  # TODO: improve


_NDArrayRunningMeanHandler = _RunningMeanHandler[np.ndarray]  # TODO: improve


@dataclasses.dataclass
class _FloatRunningMeanContainer:  # TODO: improve
    value: tp.Optional[float] = None
    buffer: _FloatRunningMeanHandler = dataclasses.field(
        default_factory=_FloatRunningMeanHandler
    )

    def flush(self) -> None:
        self.value = self.buffer.flush()


@dataclasses.dataclass
class _NDArrayRunningMeanContainer:  # TODO: improve
    value: tp.Optional[np.ndarray] = None
    buffer: _NDArrayRunningMeanHandler = dataclasses.field(
        default_factory=_NDArrayRunningMeanHandler
    )

    def flush(self) -> None:
        self.value = self.buffer.flush()


@dataclasses.dataclass(frozen=True)
class _BatchStatsRecord:  # TODO: looks like it should be rewritten before GA
    number: int
    size: int
    X_shape: tp.Tuple[int, ...]  # TODO: may be remove this?
    Y_shape: tp.Tuple[int, ...]  # TODO: may be remove this?
    loss_value: float
    lr: float
    grads: tp.Dict[str, tp.Optional[np.ndarray]]  # sub_ner_name to (int,) shaped np.array
    metrics: MetricValueContainer


@dataclasses.dataclass(frozen=True)
class _BestMoment:
    value: float
    epoch: int


class ProgressMonitor:
    def __init__(
            self,
            sub_net_names: tp.List[str],  # TODO: tp.Iterator
            metrics: MetricHandlerContainer,
            tensorboard_writer: SummaryWriter,
    ):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._metrics = metrics
        self.best_moment: tp.Optional[_BestMoment] = None
        self._tb_writer = tensorboard_writer
        self._last_finished_epoch: tp.Optional[int] = None
        self._processing_epoch: tp.Optional[int] = None

        self._last_batch_stats: tp.Dict[
                LearningMode,
                tp.Optional[_BatchStatsRecord]
        ] = dict()
        self._loss_value: tp.Dict[
                LearningMode,
                _FloatRunningMeanContainer
        ] = dict()
        self._lr: tp.Dict[
                LearningMode,
                _FloatRunningMeanContainer
        ] = dict()
        self._metric_values: tp.Dict[
                LearningMode,
                tp.Dict[str, _FloatRunningMeanContainer]
        ] = dict()
        self._grads: tp.Dict[
                LearningMode,
                tp.Dict[str, _NDArrayRunningMeanContainer]
        ] = dict()
        for mode in LearningMode:
            self._last_batch_stats[mode] = None
            self._loss_value[mode] = _FloatRunningMeanContainer()
            self._lr[mode] = _FloatRunningMeanContainer()
            self._metric_values[mode] = {
                name: _FloatRunningMeanContainer()
                for name in metrics.all
            }
            self._grads[mode] = {
                name: _NDArrayRunningMeanContainer()
                for name in sub_net_names
            }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}=("
                f"_last_finished_epoch={self._last_finished_epoch}, "
                f"_processing_epoch={self._processing_epoch}, " 
                f"_last_batch_stats={self._last_batch_stats}, "
                f"_loss_value={self._loss_value}, "
                "_grads=..."
            ")"
        )

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _enter(self) -> None:
        assert self._processing_epoch is None
        if self._last_finished_epoch is None:
            self._processing_epoch = 0
        else:
            self._processing_epoch = self._last_finished_epoch + 1
        self._logger.info(f"epoch `{self._processing_epoch}` has started")

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _exit(self, *args) -> None:  # TODO: looks like has a bug, run with few epoch num
        assert type(self._processing_epoch) is int
        for mode in LearningMode:
            self._loss_value[mode].flush()
            self._lr[mode].flush()
            for metric_name in self._metric_values[mode]:
                self._metric_values[mode][metric_name].flush()
            for sub_net_name in self._grads[mode]:
                self._grads[mode][sub_net_name].flush()
        if (
                self.best_moment is None
        ) or (
                self._metrics.all[self._metrics.main_metric_name].is_first_better_than_second(  # noqa
                    self._metric_values[LearningMode.VAL][self._metrics.main_metric_name].value,  # noqa
                    self.best_moment.value
                )
        ):
            self.best_moment: _BestMoment = _BestMoment(
                value=self._metric_values[LearningMode.VAL][self._metrics.main_metric_name].value,  # noqa
                epoch=self._processing_epoch
            )
        if self._last_finished_epoch is None:
            self._last_finished_epoch: int
        self._last_finished_epoch = self._processing_epoch
        self._processing_epoch = None
        self._logger.info(f"epoch `{self._last_finished_epoch}` has finished")

    @property
    def get_epoch_on_which_best_main_val_metric_value_got(self) -> float:
        return self._best_main_metric_value

    def log_updation(self, level: UpdationLevel, *args, **kwargs) -> None:
        if level == UpdationLevel.EPOCH:
            self._log_epoch(*args, **kwargs)
        elif level == UpdationLevel.GSTEP:
            self._log_gradient_step(*args, **kwargs)
        # TODO: elif level == UpdationLevel.ASTEP: ...
        else:
            assert False, "Unreachable line!"

    @wrap_in_logger(level="debug", ignore_args=(0, 6))
    def record_batch_processing(  # TODO: add lr tracking and embeddings
            self,
            mode: LearningMode,
            X: torch.Tensor,  # ndim = int
            Y: torch.Tensor,  # ndim = int
            Y_pred: torch.Tensor,  # ndim = int (same as Y, even shape is)
            loss: tp.Any,  # TODO: specify the type
            cntx: TrainingContext,
    ) -> None:
        assert X.shape[0] == Y.shape[0]
        assert type(self._processing_epoch) == int
        _stats = _BatchStatsRecord(
            number=0 if (
                stats := self._last_batch_stats[mode]
            ) is None else stats.number + 1,
            X_shape=X.shape,
            Y_shape=Y.shape,
            size=X.shape[0],
            loss_value=loss.cpu().item(),
            lr=cntx.optimizer.param_groups[-1]["lr"],
            grads=self._get_grads(cntx.net),
            metrics=self._metrics(
                Y.cpu().detach().numpy(),
                Y_pred.cpu().detach().numpy()
            )
        )
        self._loss_value[mode].buffer.add(_stats.loss_value, n=_stats.size)
        self._lr[mode].buffer.add(_stats.lr, n=_stats.size)
        for name, value in _stats.metrics.all.items():
            assert value is not None
            self._metric_values[mode][name].buffer.add(value, n=_stats.size)
        for sub_net_name, grad in _stats.grads.items():
            assert (grad is None) or (grad.ndim == 1)
            self._grads[mode][sub_net_name].buffer.add(grad, n=_stats.size)
        self._last_batch_stats[mode] = _stats

    def _log_epoch(self, mode: LearningMode) -> None:
        self._logger.info(f"`{mode.value}` epoch stats:")
        for sub_net_name, gradient_container in self._grads[mode].items():
            if gradient_container.value is not None:
                _norm: float = (gradient_container.value ** 2).sum() ** 0.5
                self._logger.info(f"gradnorm - {sub_net_name} - {_norm}")
                self._tb_writer.add_scalars(
                    f"gradnorm-{mode.value}/epoch",
                    {sub_net_name: _norm},
                    self._last_finished_epoch
                )
        self._logger.info(f"lr - {self._lr[mode].value}")
        self._tb_writer.add_scalar(
            f"lr-{mode.value}/epoch",
            self._lr[mode].value,
            self._last_finished_epoch
        )
        self._logger.info(f"loss - {self._loss_value[mode].value}")
        self._tb_writer.add_scalar(
            f"loss-{mode.value}/epoch",
            self._loss_value[mode].value,
            self._last_finished_epoch
        )
        for name, container in self._metric_values[mode].items():
            _msg = f"metric - {name} - {container.value}"
            if self._metrics.main_metric_name == name:
                _msg += " (main)"
            self._logger.info(_msg)
            self._tb_writer.add_scalars(
                f"metrics-{mode.value}/epoch",
                {name: container.value},
                self._last_finished_epoch
            )

    def _log_gradient_step(self, mode: LearningMode) -> None:
        self._logger.info(
            f"`{mode.value}` batch stats "
            f"({self._last_batch_stats[mode].number}-th):"
        )
        self._logger.info(f"size - {self._last_batch_stats[mode].size}")
        for sub_net_name, vec in self._last_batch_stats[mode].grads.items():  # TODO: reduce the length
            if vec is not None:
                _norm: float = (vec ** 2).sum() ** 0.5
                self._logger.info(f"gradnorm - {sub_net_name} - {_norm}")
                self._tb_writer.add_scalars(
                    f"gradnorm-{mode.value}/step",
                    {sub_net_name: _norm},
                    self._last_batch_stats[mode].number
                )
        self._logger.info(f"lr - {self._last_batch_stats[mode].lr}")
        self._tb_writer.add_scalar(
            f"lr-{mode.value}/step",
            self._last_batch_stats[mode].lr,
            self._last_batch_stats[mode].number
        )
        self._logger.info(
            f"loss - {self._last_batch_stats[mode].loss_value}"
        )
        self._tb_writer.add_scalar(
            f"loss-{mode.value}/step",
            self._last_batch_stats[mode].loss_value,
            self._last_batch_stats[mode].number
        )
        for name, value in self._last_batch_stats[mode].metrics.all.items():
            _msg = f"metric - {name} - {value}"
            if self._metrics.main_metric_name == name:
                _msg += " (main)"
            self._logger.info(_msg)
            self._tb_writer.add_scalars(
                f"metrics-{mode.value}/step",
                {name: value},
                self._last_batch_stats[mode].number
            )

    def _get_grads(
            self,
            net: torch.nn.Module
    ) -> tp.Dict[str, tp.Optional[np.ndarray]]:
        grads: tp.Dict[str, tp.Optional[np.ndarray]] = dict()  # shape: (int,)        
        for sub_net_name, sub_net in net.named_children():  # TODO: RAM memory bottle neck.
            _gradient = np.array([])
            for param in sub_net.parameters():
                grad_: tp.Optional[torch.Tensor] = param.grad
                if grad_ is not None:
                    _gradient = np.hstack(
                        (_gradient, grad_.view(-1).cpu().numpy())
                    )
            if _gradient.shape[0] > 0:
                grads[sub_net_name] = _gradient
            else:
                grads[sub_net_name] = None
        return grads

    def __del__(self):
        self._tb_writer.close()
        self._logger.info("Tensorboard writer has been closed!")
