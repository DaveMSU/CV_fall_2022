import argparse
import json
import logging
import sys

from .learning_config import LearningConfig
from .trainer import Trainer


def train_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "train",
        help=(
            "Train a model on specified data."
        )
    )
    p.set_defaults(main=train_main)

    p.add_argument(
        "-ac", "--architecture_config",
        required=True,
        type=str,
        help=(
            "POSIX path to the json file that specifies the"
            " architecture of the future neural network."
            " The config is expected to follow a schema:"
            "{"
            "    TODO: fill it"  # TODO: fill it
            "}"
        )
    )

    p.add_argument(
        "-lc", "--learning_config",
        required=True,
        type=str,
        help=(
            "POSIX path to the json file that specifies which checkpoint to"
            " use for creation of the neural network instance and that"
            " specifies the way how train it."
            " The config is expected to follow a schema:"
            "{"
            "    TODO: fill it"  # TODO:  fill it
            "}"
        )
    )

    p.add_argument(
        "--log-level",
        default="INFO",
        type=str,
        help=(
            "TODO: write me"
        )
    )


def train_main(cmd_args: argparse.Namespace) -> None:
    logging.basicConfig(
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        level=getattr(logging, cmd_args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    with open(cmd_args.architecture_config, "r") as af,\
            open(cmd_args.learning_config, "r") as lf:
        trainer = Trainer(
            net_arch_config=json.load(af),
            learning_config=LearningConfig.from_dict(json.load(lf))
        )
    trainer.run()
