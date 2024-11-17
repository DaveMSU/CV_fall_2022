import typing as tp

import dataclasses
from PIL import Image

from .. import raw_sample_pair_handlers


class FaceAndPointsResize:
    def __init__(self, size: tp.Tuple[int, int]):
        self._new_size = size

    def __call__(
            self,
            sample: raw_sample_pair_handlers.FaceAndPoints
    ) -> None:
        old_size: tp.Tuple[int, int]
        old_size, sample.input = sample.input.size, sample.input.resize(
            self._new_size,
            Image.Resampling.NEAREST
        )
        for field in dataclasses.fields(sample.output):
            coord: float = getattr(sample.output, field.name)
            setattr(
                sample.output,
                field.name,
                coord / old_size[{'x': 0, 'y': 1}[field.name[0]]]
            )
