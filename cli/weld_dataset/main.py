import argparse
import json
import pathlib

import h5py
import torch

from .datasets import (
    ImageDataset,
    ImageDatasetParams,
)
from lib import ModelInputOutputPairSample


def weld_dataset_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "weld_dataset",
        help=(
            "Creates corresponding torch datasets (at least 1)."
        )
    )
    p.set_defaults(main=weld_dataset_main)

    p.add_argument(
        "-c", "--config",
        required=True,
        type=str,
        help=(  # TODO: fix it
            "POSIX path to the json file that specifies a behaviour of"
            " welding of the datasets, the config is expected to be like:"
            "{"
            "    'datasets': ["
            "        {"
            "            'dump_path': str",
            "            'type': str",
            "            'params': {"
            "                str: tp.Any,"
            "                ..."
            "                'transforms': ["
            "                    {"
            "                        'type': str,"
            "                        'params': tp.Dict[str, tp.Any]"
            "                    },"
            "                    ..."
            "            ]"
            "        },"
            "        ..."
            "    ]"
            "}"
        )
    )


def weld_dataset_main(cmd_args: argparse.Namespace) -> None:
    with open(cmd_args.config, "r") as f:
        for d in json.load(f)["datasets"]:
            with h5py.File(pathlib.Path(d["dump_path"]), "w") as g:
                torch_ds = ImageDataset(
                    ImageDatasetParams.from_dict(d["params"])
                )
                X_shape_sample: np.Tuple[int, ...] = torch_ds[0].input.shape
                Y_shape_sample: np.Tuple[int, ...] = torch_ds[0].output.shape
                hdf5_ds_in = g.create_dataset(
                    "input",
                    (len(torch_ds), *X_shape_sample),
                    dtype=torch_ds[0].input.numpy().dtype
                )  # TODO: fix it
                hdf5_ds_out = g.create_dataset(
                    "output",
                    (len(torch_ds), *Y_shape_sample),
                    dtype=torch_ds[0].output.numpy().dtype
                )  # TODO: fix it
                for i in range(len(torch_ds)):  # TODO: speed it up?
                    sample: ModelInputOutputPairSample = torch_ds[i]
                    for first_shape, cur_field in [
                          [X_shape_sample, "input"],
                          [Y_shape_sample, "output"]
                    ]:
                        if first_shape != getattr(sample, cur_field).shape:
                            raise ValueError(
                                f"Shape of the {i}'th dataset item"
                                f" (`{cur_field}`) is"
                                f" `{getattr(sample, cur_field).shape}`"
                                f" while the first one's is `{first_shape}`,"
                                " so you have to provide the dataset welding"
                                " with a corresponding transform at least"
                            )
                    assert type(sample.input) == torch.Tensor
                    hdf5_ds_in[i] = sample.input.numpy()
                    assert type(sample.output) == torch.Tensor
                    hdf5_ds_out[i] = sample.output.numpy()
                    print(i, end="\r")  # TODO: improve logging
                print()  # TODO: improve logging
                # TODO: return: hdf5_ds.attrs["used_config"] = json.dumps(d)
