import dataclasses
import enum
import pathlib
import typing as tp


@dataclasses.dataclass(frozen=True)
class _DataloaderConfig:  # TODO: add accumulation
    dump_path: pathlib.Path
    batch_size: int
    shuffle: bool
    drop_last: bool


@dataclasses.dataclass(frozen=True)
class _DataConfig:
    train: _DataloaderConfig
    val: _DataloaderConfig


@dataclasses.dataclass(frozen=True)
class _LossConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class _OptimizerConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@enum.unique
class UpdationLevel(enum.Enum):
    EPOCH = "epoch"
    GSTEP = "gradient_step"
    # TODO: ASTEP = "accumulation_step"

    @classmethod
    def from_str(cls, s: str) -> 'UpdationLevel':
        if s == "epoch":
            return cls.EPOCH
        elif s == "gradient_step":
            return cls.GSTEP
        else:
            raise ValueError(
                f"enum.Enum type UpdationLevel doesn't maintain `{s}` entity."
            )


@dataclasses.dataclass(frozen=True)
class _LRSchedulerConfig:
    use_after: UpdationLevel
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class _HyperParamsConfig:
    loss: _LossConfig
    optimizer: _OptimizerConfig
    lr_scheduler: _LRSchedulerConfig
    total_epoch_amount: int


@dataclasses.dataclass(frozen=True)
class _SubNetOutputConfig:
    sub_net_name: str
    number_of_vectors: int
    inclusion_condition: tp.Callable[[int], bool]


@dataclasses.dataclass(frozen=True)
class _TransformOfY:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class OneMetricConfig:
    name: str
    function: str
    params: tp.Dict[str, tp.Any]  # kwargs
    target_transform: tp.Optional[_TransformOfY]
    prediction_transform: tp.Optional[_TransformOfY]

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'OneMetricConfig':
        return cls(
            name=d["name"],
            function=d["function"],
            params=d["params"],
            target_transform=_TransformOfY(**d["target_transform"]),
            prediction_transform=_TransformOfY(**d["prediction_transform"])
        )


@dataclasses.dataclass(frozen=True)
class ManyMetricsConfig:
    main: str
    all: tp.List[OneMetricConfig]


@dataclasses.dataclass(frozen=True)
class LearningConfig:
    data: _DataConfig
    hyper_params: _HyperParamsConfig
    device: str  # f.e.: "cuda:0"
    tensorboard_logs: pathlib.PosixPath
    checkpoint_dir: pathlib.PosixPath
    sub_net_outputs_to_visualize: tp.List[_SubNetOutputConfig]
    metrics: ManyMetricsConfig

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'LearningConfig':
        return cls(
            data=_DataConfig(
                train=_DataloaderConfig(
                    dump_path=pathlib.Path(d["data"]["train"]["dump_path"]),
                    batch_size=d["data"]["train"]["batch_size"],
                    shuffle=d["data"]["train"]["shuffle"],
                    drop_last=d["data"]["train"]["drop_last"]
                ),
                val=_DataloaderConfig(
                    dump_path=pathlib.Path(d["data"]["val"]["dump_path"]),
                    batch_size=d["data"]["val"]["batch_size"],
                    shuffle=d["data"]["val"]["shuffle"],
                    drop_last=d["data"]["val"]["drop_last"]
                )
            ),
            hyper_params=_HyperParamsConfig(
                loss=_LossConfig(
                    type=d["hyper_params"]["loss"]["type"],
                    params=d["hyper_params"]["loss"]["params"]
                ),
                optimizer=_OptimizerConfig(
                    type=d["hyper_params"]["optimizer"]["type"],
                    params=d["hyper_params"]["optimizer"]["params"]
                ),
                lr_scheduler=_LRSchedulerConfig(
                    use_after=UpdationLevel.from_str(
                        d["hyper_params"]["lr_scheduler"]["use_after"]
                    ),
                    type=d["hyper_params"]["lr_scheduler"]["type"],
                    params=dict(
                        map(
                            lambda k_v: (
                                k_v[0],
                                eval(k_v[1])
                            ) if k_v[0] == "lr_lambda" else k_v,
                            d["hyper_params"]["lr_scheduler"]["params"].items()  # noqa
                        )
                    )
                ),
                total_epoch_amount=d["hyper_params"]["total_epoch_amount"]
            ),
            device=d["device"],
            tensorboard_logs=pathlib.Path(d["tensorboard_logs"]),
            checkpoint_dir=pathlib.Path(d["checkpoint_dir"]),
            sub_net_outputs_to_visualize=[
                _SubNetOutputConfig(
                    sub_net_name=sub_d["sub_net_name"],
                    number_of_vectors=sub_d["number_of_vectors"],
                    inclusion_condition=eval(sub_d["inclusion_condition"])
                ) for sub_d in d["sub_net_outputs_to_visualize"]
            ],
            metrics=ManyMetricsConfig(
                main=d["metrics"]["main"],
                all=list(map(OneMetricConfig.from_dict, d["metrics"]["all"]))
            )
        )
