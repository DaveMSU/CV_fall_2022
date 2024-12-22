import random
import typing as tp

import dataclasses
from PIL import Image

from .. import raw_sample_pair_handlers


class FaceAndPointsResize:
    def __init__(self, size: tp.List[int]):
        assert len(size) == 2 and type(size[0]) is int and type(size[1]) is int  # noqa
        self._new_size: tp.Tuple[int, int] = tuple(size)

    def __call__(
            self,
            sample: raw_sample_pair_handlers.FaceAndPoints
    ) -> None:
        old_size: tp.Tuple[int, int]
        old_size, sample.input = sample.input.size, sample.input.resize(
            self._new_size,
            Image.Resampling.NEAREST
        )
        assert sample.input.size == self._new_size, (sample.input.size, self._new_size)
        for field in dataclasses.fields(sample.output):
            coord: float = getattr(sample.output, field.name)
            _id = {'x': 0, 'y': 1}[field.name[0]]
            setattr(
                sample.output,
                field.name,
                coord / old_size[_id] * self._new_size[_id]
            )


class FaceAndPointsHorizontalRandomFlip:
    def __init__(self, probability: float):
        if 0.0 <= probability <= 1.0:
            self._p = probability
        else:
            raise ValueError(
                "The probability value has to be between 0.0 and 1.0,"
                " but `{probability}` has been got"
            )

    def __call__(
            self,
            sample: raw_sample_pair_handlers.FaceAndPoints
    ) -> None:
        if (self._p == 1.0) or (random.random() < self._p):
            sample.input = sample.input.transpose(Image.FLIP_LEFT_RIGHT)
            sample.output = raw_sample_pair_handlers.FaceAndPoints.Points(
                x1=sample.input.width - sample.output.x4, y1=sample.output.y4,
                x2=sample.input.width - sample.output.x3, y2=sample.output.y3,
                x3=sample.input.width - sample.output.x2, y3=sample.output.y2,
                x4=sample.input.width - sample.output.x1, y4=sample.output.y1,
                x5=sample.input.width - sample.output.x10, y5=sample.output.y10,  # noqa
                x6=sample.input.width - sample.output.x9, y6=sample.output.y9,
                x7=sample.input.width - sample.output.x8, y7=sample.output.y8,
                x8=sample.input.width - sample.output.x7, y8=sample.output.y7,
                x9=sample.input.width - sample.output.x6, y9=sample.output.y6,
                x10=sample.input.width - sample.output.x5, y10=sample.output.y5,  # noqa
                x11=sample.input.width - sample.output.x11, y11=sample.output.y11,  # noqa
                x12=sample.input.width - sample.output.x14, y12=sample.output.y14,  # noqa
                x13=sample.input.width - sample.output.x13, y13=sample.output.y13,  # noqa
                x14=sample.input.width - sample.output.x12, y14=sample.output.y12  # noqa
            )


class FaceAndPointsMakeAbsolutePointCoordsRelative:
    def __call__(
            self,
            sample: raw_sample_pair_handlers.FaceAndPoints
    ) -> None:
        for field in dataclasses.fields(sample.output):
            coord: float = getattr(sample.output, field.name)
            setattr(
                sample.output,
                field.name,
                coord / sample.input.size[{'x': 0, 'y': 1}[field.name[0]]]
            )
