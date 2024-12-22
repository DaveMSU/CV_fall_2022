import argparse
import json

from .dataset import (
    Welder,
    WelderParams
)


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
        help=(
            "POSIX path to the json file that specifies a behaviour of"
            " welding of the datasets, the config is expected to be like:"
            "{"
            "    'datasets': ["
            "        {"
            "            'raw_x_to_raw_y_mapper': str,"
            "            'inclusion_condition': str,"
            "            'raw_model_input_output_pair_sample_type': str,"
            "            'transforms': ["
            "                {"
            "                    'type': str,"
            "                    'params': tp.Dict[str, tp.Any]"
            "                },"
            "                ..."
            "            ],"
            "            'repeat_number': int,"
            "            'dump_path': str"
            "        },"
            "        ..."
            "    ]"
            "}"
        )
    )


def weld_dataset_main(cmd_args: argparse.Namespace) -> None:
    with open(cmd_args.config, "r") as f:
        for dataset_params in json.load(f)["datasets"]:
            Welder(WelderParams.from_dict(dataset_params)).run()
