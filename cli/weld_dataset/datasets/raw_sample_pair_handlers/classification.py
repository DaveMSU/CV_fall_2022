# import dataclasses
import json
import pathlib
import typing as tp

import numpy as np
import torch
from PIL import Image

from .base import BaseRawModelInputOutputPairSample
from lib import ModelInputOutputPairSample


class ImageAndLabel(BaseRawModelInputOutputPairSample):
    @classmethod
    def create_instance(
            cls,
            input_path: pathlib.Path,
            output_path: pathlib.Path
    ) -> 'BaseRawModelInputOutputPairSample':
        with open(output_path, "r") as f:
            label: str = json.load(f)["label_of_the_class"]
        assert type(label) == str
        return cls(
            input=Image.open(input_path).convert('RGB'),
            output=label
        )

    def weld_itself(self) -> ModelInputOutputPairSample:
        assert type(self.output) == int
        return ModelInputOutputPairSample(
            torch.from_numpy(
                np.array(self.input).astype(np.float32).transpose(2, 0, 1)
            ) / 255.,
            torch.Tensor([self.output]).squeeze().long()  # TODO: is this the right shape?
        )
