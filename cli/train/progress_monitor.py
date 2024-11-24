import abc
import dataclasses
import logging
import typing as tp

import numpy as np
import torch

from .learning_config import UpdationLevel
from lib import (
    LearningMode,
    wrap_in_logger,
)

T = tp.TypeVar('T')


# TODO: want to improve this class and move it to the lib dir
#  it could be used for gradient accumulation algo, eponential running means
#  like in batch norm realisation so I will be able see several
#  running stats not for several sample amounts, it could be helpfull
#  for example when epoch is huge like in LM tasks.
class _BaseRunningMeansHandler(abc.ABC, tp.Generic[T]):
    def __init__(self):
        self._value: tp.Optional[T] = None
        self._counter: int = 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}=("
                f"_value={self._value}, "
                f"_counter={self._counter}"
            ")"
        )

    @wrap_in_logger(level="trace", ignore_args=(0,))
    def add(self, value: tp.Optional[T], n: int = 1) -> None:
        expected_type: type = self._get_type()
        self._counter += n
        if value is None:
            return
        elif type(value) is not expected_type:
            raise TypeError(
                f"The value `{value}` has to have"
                f" type `{T}`, got `{type(value)}`"
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

    @staticmethod
    @abc.abstractmethod
    def _get_type() -> type:
        raise NotImplementedError


class _FloatRunningMeansHandler(_BaseRunningMeansHandler[float]):
    # @wrap_in_logger(level="trace")  # TODO: uncomment
    @staticmethod
    def _get_type() -> type:
        return float


class _NumpyArrayRunningMeansHandler(_BaseRunningMeansHandler[np.ndarray]):
    def __repr__(self) -> str:
        vector_norm_as_its_representation = (
            self._value ** 2
        ).sum() ** 0.5 if self._value is not None else None
        return (
            f"{self.__class__.__name__}=("
                f"l2_norm: {vector_norm_as_its_representation}, "
                f"_counter={self._counter}"
            ")"
        )

    # @wrap_in_logger(level="trace")  # TODO: uncomment
    @staticmethod
    def _get_type() -> type:
        return np.ndarray


@dataclasses.dataclass(frozen=True)
class _BatchStatsRecord:  # TODO: looks like it should be rewritten before GA
    size: int
    X_shape: tp.Tuple[int, ...]  # TODO: may be remove this?
    Y_shape: tp.Tuple[int, ...]  # TODO: may be remove this?
    loss_value: float
    grads: tp.Dict[str, tp.Optional[np.ndarray]]  # sub_ner_name to (int,) shaped np.array


class ProgressMonitor:
    def __init__(self, sub_net_names: tp.List[str]):  # TODO: tp.Iterator
        self._logger = logging.getLogger(self.__class__.__name__)
        self._last_finished_epoch: tp.Optional[int] = None
        self._processing_epoch: tp.Optional[int] = None
        self._last_batch_stats: tp.Dict[
                LearningMode,
                tp.Optional[_BatchStatsRecord]
        ] = {mode: None for mode in LearningMode}
        self._loss_value = {
            "value": {
                mode: None for mode in LearningMode  # tp.Optional[float]
            },
            "buffer": {
                mode: _FloatRunningMeansHandler() for mode in LearningMode
            }
        }
        self._grads = {  # TODO: use ordered dict?
            name: {
                "value": {
                    mode: None  # tp.Optional[np.ndarray], shape: (int,)
                    for mode in LearningMode
                },
                "buffer": {
                    mode: _NumpyArrayRunningMeansHandler()
                    for mode in LearningMode
                }
            } for name in sub_net_names
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
    def __enter__(self) -> None:
        assert self._processing_epoch is None
        if self._last_finished_epoch is None:
            self._processing_epoch = 0
        else:
            self._processing_epoch = self._last_finished_epoch + 1
        self._logger.info(f"epoch `{self._processing_epoch}` has started")

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def __exit__(self, *args) -> None:  # TODO: looks like has a bug, run with few epoch num
        assert type(self._processing_epoch) == int
        for mode in LearningMode:
            self._loss_value["value"][mode] = self._loss_value[
                "buffer"
            ][mode].flush()
            for grad in self._grads.values():
                grad["value"][mode] = grad["buffer"][mode].flush()
        if self._last_finished_epoch is None:
            self._last_finished_epoch: int
        self._last_finished_epoch = self._processing_epoch
        self._processing_epoch = None
        self._logger.info(f"epoch `{self._last_finished_epoch}` has finished")

    def log_updation(self, level: UpdationLevel, *args, **kwargs) -> None:
        if level == UpdationLevel.EPOCH:
            self._log_epoch(*args, **kwargs)
        elif level == UpdationLevel.GSTEP:
            self._log_gradient_step(*args, **kwargs)
        # TODO: elif level == UpdationLevel.ASTEP: ...
        else:
            assert False, "Unreachable line!"

    @wrap_in_logger(level="debug", ignore_args=(0, 6))
    def record_batch_processing(
            self,
            mode: LearningMode,
            X: torch.Tensor,  # ndim = int
            Y: torch.Tensor,  # ndim = int
            Y_pred: torch.Tensor,  # ndim = int (same as Y, even shape is)
            loss: tp.Any,  # TODO: specify the type
            net: torch.nn.Module,  # TODO: may be specify better?
    ) -> None:
        assert X.shape[0] == Y.shape[0]
        assert type(self._processing_epoch) == int
        _stats = _BatchStatsRecord(
            X_shape=X.shape,
            Y_shape=Y.shape,
            size=X.shape[0],
            loss_value=loss.cpu().item(),
            grads=self._get_grads(net)
        )
        self._loss_value["buffer"][mode].add(_stats.loss_value, n=_stats.size)
        for sub_net_name, grad in _stats.grads.items():
            assert (grad is None) or (grad.ndim == 1)
            self._grads[sub_net_name]["buffer"][mode].add(grad, n=_stats.size)
        self._last_batch_stats[mode] = _stats

    def _log_epoch(self, mode: LearningMode) -> None:
        self._logger.info(f"`{mode.value}`\tepoch stats:")
        self._logger.info(f"loss - `{self._loss_value['value'][mode]}`")
        for sub_net_name, gradient_dict in self._grads.items():
            vec = gradient_dict["value"][mode]
            if vec is not None:
                _norm: flaot = (vec ** 2).sum() ** 0.5
                self._logger.info(f"grad norm - `{_norm}`\t(`{sub_net_name}`)")

    def _log_gradient_step(self, mode: LearningMode) -> None:
        self._logger.info(f"`{mode.value}`\tbatch stats:")
        self._logger.info(
            f"loss - `{self._last_batch_stats[mode].loss_value}`"
        )
        self._logger.info(f"size - `{self._last_batch_stats[mode].size}`")
        for sub_net_name, vec in self._last_batch_stats[mode].grads.items():  # TODO: reduce the length
            if vec is not None:
                _norm: flaot = (vec ** 2).sum() ** 0.5
                self._logger.info(f"grad norm - `{_norm}`\t(`{sub_net_name}`)")

    def _get_grads(
            self,
            net: torch.nn.Module
    ) -> tp.Dict[str, tp.Optional[np.ndarray]]:
        grads: tp.Dict[str, np.ndarray] = dict()  # shape: (int,)        
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
