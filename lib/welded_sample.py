import dataclasses
import typing as tp

import torch


@dataclasses.dataclass
class ModelInputOutputPairSample:
    input: torch.Tensor  # input for a model
    output: tp.Optional[torch.Tensor]  # the expected output

    def __init__(self, input_, output_, /):
        self.input, self.output = input_, output_
        self._validate_types()

    def _validate_types(self):
        assert type(self.input) == torch.Tensor
        assert (self.output is None) or (type(self.output) == torch.Tensor)
