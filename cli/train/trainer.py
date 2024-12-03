import typing as tp

import torch
from torch.utils.tensorboard import SummaryWriter

from .dataset import HDF5Dataset
from .learning_config import LearningConfig, UpdationLevel
from .metric_factory import MetricHandlerContainer
from .net_factory import NetFactory
from .progress_monitor import ProgressMonitor
from lib import (
    LearningMode,
    ModelInputOutputPairSample,  # TODO: use it or delete from here
    wrap_in_logger,
)


class TrainingContext:  # TODO: deal with _attrs
    @wrap_in_logger(level="debug", ignore_args=(0,))
    def __init__(
            self,
            net_arch_config: tp.Dict[str, tp.Any],  # result of json.load(*)
            learning_config: LearningConfig
    ):
        self._init_dataloaders(learning_config)
        self._init_device(learning_config)
        self._init_net(net_arch_config)
        self._init_hyper_params(learning_config)
        self._init_monitorings(learning_config)

    def __repr__(self) -> str:
        return (
            "TrainingContext("
                f"dataloaders={self._dataloaders}, "
                f"device={self._device}, "
                "net=torch.nn.Module, "
                f"loss={self._loss}, "
                f"lr_scheduler={self._lr_scheduler}, "
                f"total_epoch_amount={self._total_epoch_amount}, "
                f"writer={self._writer}"
            ")"
         )

    @property
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def dataloaders(self) -> tp.Dict[
            LearningMode,
            torch.utils.data.DataLoader
    ]:
        return self._dataloaders

    @property
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def device(self) -> torch.device:
        return self._device

    @property
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def net(self) -> torch.nn.Module:
        return self._net

    @property
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def loss(self) -> torch.nn.modules.loss._Loss:
        return self._loss

    @property
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    @wrap_in_logger(level="debug", ignore_args=(0,))
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._lr_scheduler

    @property
    @wrap_in_logger(level="debug", ignore_args=(0,))
    def total_epoch_amount(self) -> int:
        return self._total_epoch_amount

    @property
    @wrap_in_logger(level="debug", ignore_args=(0,))  # TODO: may be 'trace'?
    def writer(self) -> SummaryWriter:
        return self._writer

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _init_dataloaders(self, learning_config: LearningConfig) -> None:
        self._dataloaders: tp.Dict[
                LearningMode,
                torch.utils.data.DataLoader
        ] = dict()
        for mode in LearningMode:
            self._dataloaders[mode] = torch.utils.data.DataLoader(
                HDF5Dataset(
                    getattr(learning_config.data, mode.value).dump_path
                ),
                batch_size=getattr(
                    learning_config.data,
                    mode.value
                ).batch_size,
                shuffle=getattr(learning_config.data, mode.value).shuffle
            )

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _init_device(self, learning_config: LearningConfig) -> None:  # TODO: make it a list of devices instead
        self._device = torch.device(learning_config.device)

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _init_net(self, net_arch_config: tp.Dict[str, tp.Any]) -> None:
        self._net = NetFactory.create_network(net_arch_config)
        self._net = self._net.to(self._device)
        
    @wrap_in_logger(level="debug", ignore_args=(0,))
    def _init_hyper_params(self, learning_config: LearningConfig) -> None:
        self._loss = getattr(
            torch.nn,
            learning_config.hyper_params.loss.type
        )(
            **learning_config.hyper_params.loss.params
        )
        self._optimizer = getattr(
            torch.optim,
            learning_config.hyper_params.optimizer.type
        )(
            **{
                "params": self._net.parameters(),
                **learning_config.hyper_params.optimizer.params
            }
        )
        self._lr_scheduler = getattr(
            torch.optim.lr_scheduler,
            learning_config.hyper_params.lr_scheduler.type
        )(
            **{
                "optimizer": self._optimizer,
                **learning_config.hyper_params.lr_scheduler.params
            }
        )
        if (tea := learning_config.hyper_params.total_epoch_amount) < 0:
            raise ValueError(
                f"Total epoch amount must be non-negative, but {tea} occured"
            )
        else:
            self._total_epoch_amount: int = tea

    @wrap_in_logger(level="debug", ignore_args=(0,))  # TODO: rm next lines
    def _init_monitorings(self, learning_config: LearningConfig) -> None:
        pass
        # self._writer = SummaryWriter(learning_config.tensorboard_logs)  # TODO: move and this behivour to PMonitor
        # self._metrics = MetricContainer.from_config(learning_config.metrics)


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
            mode, X, Y, Y_pred, loss_value, self._cntx.net
        )
