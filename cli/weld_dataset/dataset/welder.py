import dataclasses
import pathlib
import typing as tp

import h5py
import numpy as np
import pandas as pd
import torch

from . import raw_sample_pair_handlers
from lib import ModelInputOutputPairSample
from . import transforms


NUMPY_SEED = 7  # TODO; thing about to create separate dir for such vars


@dataclasses.dataclass(frozen=True)
class WelderParams:
    @dataclasses.dataclass(frozen=True)
    class RawSampleId:
        x: pathlib.Path
        y: pathlib.Path
    raw_x_to_raw_y_mapper: tp.List[RawSampleId]

    inclusion_condition: tp.Callable[[str], bool]
    RawModelInputOutputPairSample: type
    transforms: tp.Tuple[
        tp.Callable[
            [raw_sample_pair_handlers.BaseRawModelInputOutputPairSample],
            None
        ]
    ]
    repeat_number: int
    dump_path: pathlib.PosixPath

    @classmethod
    def from_dict(cls, d: tp.Dict[str, tp.Any]) -> 'WelderParams':
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
            ),
            repeat_number=d["repeat_number"],
            dump_path=pathlib.Path(d["dump_path"])
        )


class Welder:
    def __init__(self, params: WelderParams):
        self._sample_ids: tp.List[WelderParams.RawSampleId] = list(
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
        self._repeat_number: int = params.repeat_number
        self._dump_path: pathlib.PosixPath = params.dump_path

    def run(self) -> None:
        with h5py.File(pathlib.Path(self._dump_path), "w") as g:
            _welded_sample = self._get_welded_raw_sample(0)
            X_shape_sample: np.Tuple[int, ...] = _welded_sample.input.shape
            Y_shape_sample: np.Tuple[int, ...] = _welded_sample.output.shape
            hdf5_ds_in = g.create_dataset(
                "input",
                (self._total_number_of_samples, *X_shape_sample),
                dtype=_welded_sample.input.numpy().dtype
            )  # TODO: fix it
            hdf5_ds_out = g.create_dataset(
                "output",
                (self._total_number_of_samples, *Y_shape_sample),
                dtype=_welded_sample.output.numpy().dtype
            )  # TODO: fix it
            outer_i: int = 0
            for lap in range(self._repeat_number):
                for inner_i in range(len(self._sample_ids)):  # TODO: speedup?
                    assert outer_i == len(self._sample_ids) * lap + inner_i
                    sample = self._get_welded_raw_sample(inner_i)
                    for first_shape, cur_field in [
                          [X_shape_sample, "input"],
                          [Y_shape_sample, "output"]
                    ]:
                        if first_shape != getattr(sample, cur_field).shape:
                            raise ValueError(
                                f"Shape of the {inner_i}'th dataset item"
                                f" (`{cur_field}`) is"
                                f" `{getattr(sample, cur_field).shape}`"
                                f" while the first one's is `{first_shape}`,"
                                " so you have to provide the dataset welding"
                                " with a corresponding transform at least"
                            )
                    else:
                        assert type(sample.input) is torch.Tensor
                        hdf5_ds_in[outer_i] = sample.input.numpy()
                        assert type(sample.output) is torch.Tensor
                        hdf5_ds_out[outer_i] = sample.output.numpy()
                        print(outer_i, end="\r")  # TODO: improve logging
                        outer_i += 1
            print()  # TODO: improve logging
            # TODO: return: hdf5_ds.attrs["used_config"] = json.dumps(d)

    @property
    def _total_number_of_samples(self):
        return len(self._sample_ids) * self._repeat_number

    def _get_welded_raw_sample(self, index: int) -> ModelInputOutputPairSample:
        raw_sample = self._raw_sample_factory(
            input_path=self._sample_ids[index].x,
            output_path=self._sample_ids[index].y
        )
        for current_transform in self._transforms:
            current_transform(raw_sample)
        return raw_sample.weld_itself()
