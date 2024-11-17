import typing as tp

import torch
from torch.utils.tensorboard import SummaryWriter

from .dataset import HDF5Dataset
from .learning_config import LearningConfig
from .net_factory import NetFactory


class TrainingContext:  # TODO: deal with _attrs
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

    @property
    def dataloaders(self) -> tp.Dict[str, torch.utils.data.DataLoader]:
        return self._dataloaders

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def net(self) -> torch.nn.Module:
        return self._net

    @property
    def loss(self) -> torch.nn.modules.loss._Loss:
        return self._loss

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        return self._lr_scheduler

    @property
    def total_epoch_amount(self) -> int:
        return self._total_epoch_amount

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    def _init_dataloaders(self, learning_config: LearningConfig) -> None:
        self._dataloaders: tp.Dict[str, DataLoader] = dict()
        for mode in ["train", "val"]:
            self._dataloaders[mode] = torch.utils.data.DataLoader(
                HDF5Dataset(getattr(learning_config.data, mode).dump_path),
                batch_size=getattr(learning_config.data, mode).batch_size,
                shuffle=getattr(learning_config.data, mode).shuffle
            )

    def _init_device(self, learning_config: LearningConfig) -> None:  # TODO: make it a list of devices instead
        self._device = torch.device(learning_config.device)

    def _init_net(self, net_arch_config: tp.Dict[str, tp.Any]) -> None:
        self._net = NetFactory.create_network(net_arch_config)
        self._net = self._net.to(self._device)
        
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

    def _init_monitorings(self, learning_config: LearningConfig) -> None:
        self._writer = SummaryWriter(learning_config.tensorboard_logs)
        self._metrics = tp.Any  # TODO: implement


class Trainer:  # TODO: make it a singleton
    def __init__(
            self,
            net_arch_config: tp.Dict[str, tp.Any],  # result of json.load(*)
            learning_config: LearningConfig
    ):
        self._cntx = TrainingContext(net_arch_config, learning_config)

    def run(self) -> None:
        for epoch in range(self._cntx.total_epoch_amount):
            self._train_once_on_the_dataset(epoch)
            self._validate_once_on_the_dataset(epoch)

    def _train_once_on_the_dataset(self, epoch: int) -> None:
        import numpy as np
        train_loss_history = []
        grads_norm_history = []
        self._cntx.net.train()
        for X, y in self._cntx.dataloaders["train"]:
            self._cntx.optimizer.zero_grad()
            X = X.to(self._cntx.device)
            y = y.to(self._cntx.device)
            y_pred = self._cntx.net(X)
            loss_value = self._cntx.loss(y_pred, y)
            # print("train loss:", loss_value.cpu().item())
            loss_value.backward()
            self._cntx.optimizer.step()
            train_loss_history.append(loss_value.cpu().item())
            grad = []
            for param in self._cntx.net.parameters():
                if param.grad is not None:
                    grad.extend(param.grad.cpu().reshape(-1).tolist())
            grads_norm_history.append((np.array(grad) ** 2).sum() ** 0.5)
        self._cntx.optimizer.zero_grad()
        print("mean train loss:", sum(train_loss_history) / len(train_loss_history))
        print("grad norm:", np.mean(grads_norm_history))

    def _validate_once_on_the_dataset(self, epoch: int) -> None:
        self._cntx.net.eval()
        for X, y in self._cntx.dataloaders["val"]:
            X = X.to(self._cntx.device)
            y = y.to(self._cntx.device)
            y_pred = self._cntx.net(X)
            loss_value = self._cntx.loss(y_pred, y)
            print("val loss:", loss_value.cpu().item())
