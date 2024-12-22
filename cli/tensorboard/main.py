import argparse
import os

from .app import tensorboard_app


def tensorboard_add_cmdargs(
        parser: argparse.ArgumentParser,
        subparsers: argparse._SubParsersAction
) -> None:
    p = subparsers.add_parser(
        "tensorboard",
        help="Run tensorboard app, based on flask server"
    )
    p.set_defaults(main=tensorboard_main)


def tensorboard_main(cmd_args: argparse.Namespace) -> None:
    tensorboard_app.run(
        host=os.getenv("LAN_HOST_IP"),
        port=os.getenv("FLASK_PORT"),
        debug=True
    )
