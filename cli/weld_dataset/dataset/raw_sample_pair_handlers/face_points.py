import dataclasses
import json
import pathlib
import typing as tp

import numpy as np
import torch
from PIL import Image

from .base import BaseRawModelInputOutputPairSample
from lib import ModelInputOutputPairSample


class FaceAndPoints(BaseRawModelInputOutputPairSample):
    @dataclasses.dataclass  # TODO: inherit, not re-specify
    class Points:
        x1: float
        y1: float
        x2: float
        y2: float
        x3: float
        y3: float
        x4: float
        y4: float
        x5: float
        y5: float
        x6: float
        y6: float
        x7: float
        y7: float
        x8: float
        y8: float
        x9: float
        y9: float
        x10: float
        y10: float
        x11: float
        y11: float
        x12: float
        y12: float
        x13: float
        y13: float
        x14: float
        y14: float
    output: Points

    @classmethod
    def create_instance(
            cls,
            input_path: pathlib.Path,
            output_path: pathlib.Path
    ) -> 'BaseRawModelInputOutputPairSample':
        with open(output_path, "r") as f:
            raw_points: tp.Dict[str, float] = json.load(f)
        return cls(
            input=Image.open(input_path).convert('RGB'),
            output=cls.Points(**raw_points)
        )

    def weld_itself(self) -> ModelInputOutputPairSample:
        return ModelInputOutputPairSample(
            torch.from_numpy(
                np.array(self.input).astype(np.float32).transpose(2, 0, 1)
            ) / 255.,
            torch.Tensor(
                [
                    getattr(self.output, k) for k in [
                        "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                        "x5", "y5", "x6", "y6", "x7", "y7", "x8", "y8",
                        "x9", "y9",
                        "x10", "y10", "x11", "y11", "x12", "y12",
                        "x13", "y13", "x14", "y14"
                    ]
                ]
            )
        )
