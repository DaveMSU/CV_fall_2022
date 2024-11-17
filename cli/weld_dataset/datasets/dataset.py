import dataclasses
import pathlib
import typing as tp

import numpy as np
import pandas as pd
import torch

from . import raw_sample_pair_handlers
# from .welded_sample_pair_handler import ModelInputOutputPairSample
from lib import ModelInputOutputPairSample
from . import transforms


NUMPY_SEED = 7  # TODO; thing about to create separate dir for such vars


@dataclasses.dataclass(frozen=True)
class ImageDatasetParams:
    @dataclasses.dataclass(frozen=True)
    class RawSampleId:
        x: pathlib.Path
        y: pathlib.Path
    raw_x_to_raw_y_mapper: tp.List[RawSampleId]

    inclusion_condition: tp.Callable[[str], bool]
    RawModelInputOutputPairSample: type  # TODO: maybe use factory function instead, 'type' is too wide?
    transforms: tp.Tuple[
        tp.Callable[
            [raw_sample_pair_handlers.BaseRawModelInputOutputPairSample],
            None
        ]
    ]
    
    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'ImageDatasetParams':
        return cls(
            raw_x_to_raw_y_mapper=[
                cls.RawSampleId(
                    x=pathlib.Path(r[1].x_path),
                    y=pathlib.Path(r[1].y_path)
                ) for r in pd.read_csv(d["raw_x_to_raw_y_mapper"]).iterrows()
            ],
            inclusion_condition=eval(d["inclusion_condition"]),
            RawModelInputOutputPairSample=getattr(
                raw_sample_pair_handlers,
                d["raw_model_input_output_pair_sample_type"]
            ),
            transforms=tuple(
                [lambda x: x] if d["transforms"] is None else [
                    getattr(transforms, sub_d["type"])(**sub_d["params"])
                    for sub_d in d["transforms"]
                ]
            )
        )


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, params: ImageDatasetParams):
        self._sample_ids: tp.List[RawSampleId] = list(
            filter(params.inclusion_condition, params.raw_x_to_raw_y_mapper)
        )
        
        for raw_sample_id in self._sample_ids:
            for field in dataclasses.fields(raw_sample_id):
                if not (path := getattr(raw_sample_id, field.name)).exists():
                    raise FileNotFoundError(f"File `{path}` doesn't exist.")

        np.random.seed(NUMPY_SEED)
        permuted_indexes = np.random.permutation(len(self._sample_ids))
        self._sample_ids = [self._sample_ids[i] for i in permuted_indexes]
        self._raw_sample_factory: tp.Callable[
            [pathlib.Path, pathlib.Path],
            raw_sample_pair_handlers.BaseRawModelInputOutputPairSample
        ] = params.RawModelInputOutputPairSample.create_instance
        self._transforms: torch.nn.Sequential = params.transforms
        
    # @property
    # def shape(self) -> tp.Tuple[int, int]:
    #     return (self._new_shape.height, self._new_shape.width)

    def __len__(self):
        return len(self._sample_ids)

    def __getitem__(self, index: int) -> ModelInputOutputPairSample:
        raw_sample = self._raw_sample_factory(
            input_path=self._sample_ids[index].x,
            output_path=self._sample_ids[index].y
        )
        for current_transform in self._transforms:
            current_transform(raw_sample)
        return raw_sample.weld_itself()
