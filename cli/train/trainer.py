# import pathlib
import typing as tp

import torch
from torch.utils.tensorboard import SummaryWriter

# from .dataset import HDF5Dataset
from .learning_config import LearningConfig, UpdationLevel
from .metric_factory import MetricHandlerContainer
# from .net_factory import NetFactory
from .progress_monitor import ProgressMonitor
from .training_context import TrainingContext
from lib import (
    LearningMode,
    wrap_in_logger,
)


class Trainer:  # TODO: make it a singleton
    def __init__(
            self,
            net_arch_config: tp.Dict[str, tp.Any],  # result of json.load(*)
            learning_config: LearningConfig
    ):
        self._cntx = TrainingContext(net_arch_config, learning_config)
        self._progress_monitor = ProgressMonitor(
            sub_net_names=[
                x[0] for x in self._cntx.net.named_children()
            ],
            metrics=MetricHandlerContainer.from_config(
                learning_config.metrics
            ),
            tensorboard_writer=SummaryWriter(
                learning_config.tensorboard_logs
            )
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
                "TrainingContext=..., "
                f"ProgressMonitor={self._progress_monitor}"
            ")"
        )

    @wrap_in_logger(level="debug")
    def run(self) -> None:  # TODO: return here after Dima's code review
        for epoch in range(self._cntx.total_epoch_amount):
            # with self._progress_monitor:  # TODO: re-think what epoch is; upd: it also causes ambuguity when expcetion is raised inside of the context manager
            self._progress_monitor._enter()
            self._process_dataset(LearningMode.TRAIN)
            self._process_dataset(LearningMode.VAL)
            self._progress_monitor._exit()
            self._progress_monitor.log_updation(UpdationLevel.EPOCH, LearningMode.TRAIN)  # TODO: reduce the length
            self._progress_monitor.log_updation(UpdationLevel.EPOCH, LearningMode.VAL)  # TODO: reduce the length
            if self._progress_monitor.best_moment.epoch == epoch:
                self._cntx.save_checkpoint("best")
            self._cntx.save_checkpoint("last")

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _process_dataset(self, mode: LearningMode) -> None:
        getattr(
            self._cntx.net,
            {LearningMode.TRAIN: "train", LearningMode.VAL: "eval"}[mode]
        )()
        for X, Y in self._cntx.dataloaders[mode]:
            if mode == LearningMode.TRAIN:
                self._cntx.optimizer.zero_grad()
            self._process_batch(X, Y, mode)
            if mode == LearningMode.TRAIN:
                self._cntx.optimizer.step()
                self._cntx.optimizer.zero_grad()
            self._progress_monitor.log_updation(UpdationLevel.GSTEP, mode)  # TODO: may be it should be moved after GA maintainance

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _process_batch(
            self,
            X: torch.Tensor,
            Y: torch.Tensor,
            /,
            mode: LearningMode
    ) -> None:
        X = X.to(self._cntx.device)
        Y = Y.to(self._cntx.device)
        Y_pred = self._cntx.net(X)
        loss_value = self._cntx.loss(Y_pred, Y)
        if mode == LearningMode.TRAIN:
            loss_value.backward()
        self._progress_monitor.record_batch_processing(
            mode, X, Y, Y_pred, loss_value, self._cntx
        )
