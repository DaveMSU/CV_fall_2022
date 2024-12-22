from flask import Flask

tensorboard_app = Flask(__name__)
from . import routes  # noqa: E402


__all__ = [
    "routes",
    "tensorboard_app"
]
