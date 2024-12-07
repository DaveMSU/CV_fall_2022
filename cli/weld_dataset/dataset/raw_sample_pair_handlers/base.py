import abc
import dataclasses
import pathlib
import typing as tp

from lib import ModelInputOutputPairSample


@dataclasses.dataclass  # mutable!
class BaseRawModelInputOutputPairSample(abc.ABC):
    input: tp.Any  # TODO: better typing
    output: tp.Any  # TODO: better typing

    @classmethod  # TODO: use metatype
    @abc.abstractmethod
    def create_instance(
            cls,
            input_path: pathlib.Path,
            output_path: pathlib.Path
    ) -> 'BaseRawModelInputOutputPairSample':
        raise NotImplementedError

    @abc.abstractmethod
    def weld_itself(
            self
    ) -> ModelInputOutputPairSample:
        raise NotImplementedError
