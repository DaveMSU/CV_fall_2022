import dataclasses
import enum
import pathlib
import typing as tp


@dataclasses.dataclass(frozen=True)
class _LoggerConfig:
    logger_name: str
    logging_file: pathlib.Path
    stdout: bool


@dataclasses.dataclass(frozen=True)
class _DataloaderConfig:  # TODO: add accumulation
    dump_path: pathlib.Path
    batch_size: int
    shuffle: bool


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
            ValueError(
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
class _ModelDumpsConfig:
    best: pathlib.Path
    last: pathlib.Path


@dataclasses.dataclass(frozen=True)
class _SubNetOutputConfig:
    sub_net_name: str
    number_of_vectors: int
    inclusion_condition: tp.Callable[[int], bool]


@dataclasses.dataclass(frozen=True)
class _OneMetricConfig:
    name: str
    function: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class _ManyMetricsConfig:
    main: str
    all: tp.List[_OneMetricConfig]


@dataclasses.dataclass(frozen=True)
class LearningConfig:
    logger: _LoggerConfig
    data: _DataConfig
    continue_from: tp.Optional[pathlib.Path]
    hyper_params: _HyperParamsConfig
    device: str  # f.e.: "cuda:{i}"
    tensorboard_logs: pathlib.Path
    model_dumps: _ModelDumpsConfig
    sub_net_outputs_to_visualize: tp.List[_SubNetOutputConfig]
    metrics: _ManyMetricsConfig

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'LearningConfig':
        return cls(
            logger=_LoggerConfig(
                logger_name=d["logger"]["logger_name"],
                logging_file=pathlib.Path(d["logger"]["logging_file"]),
                stdout=d["logger"]["stdout"]
            ),
            data=_DataConfig(
                train=_DataloaderConfig(
                    dump_path=pathlib.Path(d["data"]["train"]["dump_path"]),
                    batch_size=d["data"]["train"]["batch_size"],
                    shuffle=d["data"]["train"]["shuffle"]
                ),
                val=_DataloaderConfig(
                    dump_path=pathlib.Path(d["data"]["val"]["dump_path"]),
                    batch_size=d["data"]["val"]["batch_size"],
                    shuffle=d["data"]["val"]["shuffle"]
                )
            ),
            continue_from=d["continue_from"],
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
            model_dumps=_ModelDumpsConfig(
                best=pathlib.Path(d["model_dumps"]["best"]),
                last=pathlib.Path(d["model_dumps"]["last"])
            ),
            sub_net_outputs_to_visualize=[
                _SubNetOutputConfig(
                    sub_net_name=sub_d["sub_net_name"],
                    number_of_vectors=sub_d["number_of_vectors"],
                    inclusion_condition=eval(sub_d["inclusion_condition"])
                ) for sub_d in d["sub_net_outputs_to_visualize"]
            ],
            metrics=_ManyMetricsConfig(
                main=d["metrics"]["main"],
                all=[
                    _OneMetricConfig(**sub_d) for sub_d in d["metrics"]["all"]
                ]
            )
        )
