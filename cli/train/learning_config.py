import dataclasses
import pathlib
import typing as tp
from enum import Enum

@dataclasses.dataclass(frozen=True)
class LoggerConfig:
    logger_name: str
    logging_file: pathlib.Path
    stdout: bool


@dataclasses.dataclass(frozen=True)
class DataloaderConfig:  # TODO: add accumulation
    dump_path: pathlib.Path
    batch_size: int
    shuffle: bool


@dataclasses.dataclass(frozen=True)
class DataConfig:
    train: DataloaderConfig
    val: DataloaderConfig


@dataclasses.dataclass(frozen=True)
class LossConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class OptimizerConfig:
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


class UpdationLevel(Enum):
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
                f"Enum type UpdationLevel doesn't maintain `{s}` entity."
            )


@dataclasses.dataclass(frozen=True)
class LRSchedulerConfig:
    use_after: UpdationLevel
    type: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class HyperParamsConfig:
    loss: LossConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    total_epoch_amount: int


@dataclasses.dataclass(frozen=True)
class ModelDumpsConfig:
    best: pathlib.Path
    last: pathlib.Path


@dataclasses.dataclass(frozen=True)
class SubNetOutputConfig:
    sub_net_name: str
    number_of_vectors: int
    inclusion_condition: tp.Callable[[int], bool]


@dataclasses.dataclass(frozen=True)
class OneMetricConfig:
    name: str
    function: str
    params: tp.Dict[str, tp.Any]  # kwargs


@dataclasses.dataclass(frozen=True)
class ManyMetricsConfig:
    main: str
    all: tp.List[OneMetricConfig]


@dataclasses.dataclass(frozen=True)
class LearningConfig:
    logger: LoggerConfig
    data: DataConfig
    continue_from: tp.Optional[pathlib.Path]
    hyper_params: HyperParamsConfig
    device: str  # f.e.: "cuda:{i}"
    tensorboard_logs: pathlib.Path
    model_dumps: ModelDumpsConfig
    sub_net_outputs_to_visualize: tp.List[SubNetOutputConfig]
    metrics: ManyMetricsConfig

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'LearningConfig':
        return cls(
            logger=LoggerConfig(
                logger_name=d["logger"]["logger_name"],
                logging_file=pathlib.Path(d["logger"]["logging_file"]),
                stdout=d["logger"]["stdout"]
            ),
            data=DataConfig(
                train=DataloaderConfig(
                    dump_path=pathlib.Path(d["data"]["train"]["dump_path"]),
                    batch_size=d["data"]["train"]["batch_size"],
                    shuffle=d["data"]["train"]["shuffle"]
                ),
                val=DataloaderConfig(
                    dump_path=pathlib.Path(d["data"]["val"]["dump_path"]),
                    batch_size=d["data"]["val"]["batch_size"],
                    shuffle=d["data"]["val"]["shuffle"]
                )
            ),
            continue_from=d["continue_from"],
            hyper_params=HyperParamsConfig(
                loss=LossConfig(
                    type=d["hyper_params"]["loss"]["type"],
                    params=d["hyper_params"]["loss"]["params"]
                ),
                optimizer=OptimizerConfig(
                    type=d["hyper_params"]["optimizer"]["type"],
                    params=d["hyper_params"]["optimizer"]["params"]
                ),
                lr_scheduler=LRSchedulerConfig(
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
            model_dumps=ModelDumpsConfig(
                best=pathlib.Path(d["model_dumps"]["best"]),
                last=pathlib.Path(d["model_dumps"]["last"])
            ),
            sub_net_outputs_to_visualize=[
                SubNetOutputConfig(
                    sub_net_name=sub_d["sub_net_name"],
                    number_of_vectors=sub_d["number_of_vectors"],
                    inclusion_condition=eval(sub_d["inclusion_condition"])
                ) for sub_d in d["sub_net_outputs_to_visualize"]
            ],
            metrics=ManyMetricsConfig(
                main=d["metrics"]["main"],
                all=[
                    OneMetricConfig(**sub_d) for sub_d in d["metrics"]["all"]
                ]
            )
        )
