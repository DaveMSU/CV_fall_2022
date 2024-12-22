import typing as tp

import numpy as np

from .. import raw_sample_pair_handlers


class ModelOutputStrToIntMapper:
    def __init__(self, mapper: tp.Dict[str, int]):
        self._mapper = dict(**mapper)

    def __call__(
            self,
            sample: raw_sample_pair_handlers.BaseRawModelInputOutputPairSample
    ) -> None:
        if type(sample.output) is not str:
            raise ValueError(
                "The str was expected for sample's output as a type,"
                f" but `{type(sample.output)}` has occured instead"
            )
        else:
            sample.output = self._mapper[sample.output]


class ModelOutputIntToOneHotMaker:
    def __init__(self, amount_of_classes: int):
        assert amount_of_classes >= 2
        self._amount_of_classes = amount_of_classes

    def __call__(
            self,
            sample: raw_sample_pair_handlers.BaseRawModelInputOutputPairSample
    ) -> None:
        one_hot = np.zeros(self._amount_of_classes, dtype=np.float32)
        one_hot[sample.output] = 1.0
        sample.output = one_hot
