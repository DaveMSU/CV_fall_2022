from flask import Flask

tensorboard_app = Flask(__name__)
from . import routes

