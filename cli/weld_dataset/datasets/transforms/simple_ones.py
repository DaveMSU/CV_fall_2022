import typing as tp

from .. import raw_sample_pair_handlers


class ModelOutputStrToIntMapper:
    def __init__(self, mapper: tp.Dict[str, int]):
        self._mapper = dict(**mapper)
    
    def __call__(
            self,
            sample: raw_sample_pair_handlers.BaseRawModelInputOutputPairSample
    ) -> None:
        if type(sample.output) != str:
            raise ValueError(
                "The str was expected for sample's output as a type,"
                f" but `{type(sample.output)}` has occured instead"
            )
        else:
            sample.output = self._mapper[sample.output]
