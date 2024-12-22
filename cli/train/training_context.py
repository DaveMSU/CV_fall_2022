import pathlib
import typing as tp

import torch
from torch.utils.tensorboard import SummaryWriter

from .dataset import HDF5Dataset
from .learning_config import LearningConfig, UpdationLevel
from .net_factory import NetFactory
from lib import (
    LearningMode,
    wrap_in_logger,
)


class _LRSchedulerWrapper:
    def __init__(
            self,
            lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
            *,
            react_only_on: UpdationLevel
    ):
        self._lr_scheduler = lr_scheduler
        self._updation_level_to_react_on = react_only_on

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def step(
            self,
            updation_level: UpdationLevel,
            loss_value: float,
            mode: LearningMode
    ) -> None:
        if updation_level == self._updation_level_to_react_on:
            if type(self._lr_scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:  # noqa
                if mode == LearningMode.VAL:
                    self._lr_scheduler.step(loss_value)
            else:
                if mode == LearningMode.TRAIN:
                    self._lr_scheduler.step()


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
        self._init_checkpoint_settings(learning_config)

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
    def lr_scheduler(self) -> _LRSchedulerWrapper:
        return self._lr_scheduler_wrapper

    @property
    @wrap_in_logger(level="debug", ignore_args=(0,))
    def total_epoch_amount(self) -> int:
        return self._total_epoch_amount

    @property
    @wrap_in_logger(level="debug", ignore_args=(0,))  # TODO: may be 'trace'?
    def writer(self) -> SummaryWriter:
        return self._writer

    @property
    @wrap_in_logger(level="debug", ignore_args=(0,))  # TODO: may be 'trace'?
    def checkpoint_dir(self) -> pathlib.PosixPath:
        return self._checkpoint_dir

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def save_checkpoint(self, basename: str) -> None:
        torch.save(
            {
                "model_state_dict": self._net.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "loss_state_dict": self._loss.state_dict(),
                "lr_scheduler_state_dict": self._lr_scheduler.state_dict(),
                "device": self._device,
            },
            self._checkpoint_dir / f"{basename}.pth"
        )

    def load_checkpoint(self, path: pathlib.PosixPath) -> None:
        checkpoint = torch.load(path, map_location=self._device)
        for attr in ["net", "optimizer", "loss", "lr_scheduler"]:
            getattr(self, f"_{attr}").load_state_dict(
                checkpoint[f"{attr}_state_dict"]
            )

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
                shuffle=getattr(learning_config.data, mode.value).shuffle,
                drop_last=getattr(learning_config.data, mode.value).drop_last
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
        self._lr_scheduler_wrapper = _LRSchedulerWrapper(
            self._lr_scheduler,
            react_only_on=learning_config.hyper_params.lr_scheduler.use_after
        )
        if (tea := learning_config.hyper_params.total_epoch_amount) < 0:
            raise ValueError(
                f"Total epoch amount must be non-negative, but {tea} occured"
            )
        else:
            self._total_epoch_amount: int = tea

    @wrap_in_logger(level="debug", ignore_args=(0,))  # TODO: rm next lines
    def _init_checkpoint_settings(
            self,
            learning_config: LearningConfig
    ) -> None:
        learning_config.checkpoint_dir.mkdir(parents=False, exist_ok=True)
        self._checkpoint_dir = learning_config.checkpoint_dir
