import abc
import typing as tp

import numpy as np

from lib import wrap_in_logger


Output = tp.TypeVar('T')


class BaseTransform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, array: Output) -> Output:
        assert isinstance(array, np.ndarray)
        raise NotImplementedError


class Identical(BaseTransform):
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def __call__(self, array: Output) -> Output:
        return array


class Argmax(BaseTransform):
    @wrap_in_logger(level="trace", ignore_args=(0,))
    def __init__(self, axis: int):
        self._axis: int = axis

    @wrap_in_logger(level="trace", ignore_args=(0,))
    def __call__(self, array: Output) -> Output:
        return array.argmax(axis=self._axis)


class SoftmaxAlongAxis0(BaseTransform):
    @wrap_in_logger(level="debug", ignore_args=(0,))
    def __init__(self, temperature: float):
        self._t: float = temperature

    @wrap_in_logger(level="debug", ignore_args=(0,))
    def __call__(self, array: Output) -> Output:
        assert isinstance(array, np.ndarray)
        _dim_to_reduce = tuple(range(1, array.ndim))
        exp_array = np.exp(
            np.clip(
                array / self._t,
                a_min=-8.0e+307,
                a_max=705.0
            )
        )
        return np.clip(
            exp_array / exp_array.sum(axis=_dim_to_reduce, keepdims=True),
            a_min=1e-63,
            a_max=np.float64("+inf")
        )
