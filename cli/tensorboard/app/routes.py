import os
import pathlib
import signal

import flask
import requests
import tensorboard
from multiprocessing import Process

from . import tensorboard_app


PREFIX_PATH = pathlib.Path("/var/lib/storage/resources/experiments")  # TODO: param?
SERVER_PROTOCOL: str = "http"  # TODO: try to move app to https
LAN_HOST_IP: str = os.environ["LAN_HOST_IP"]
WAN_HOST_IP: str = os.environ["WAN_HOST_IP"]
TENSORBOARD_PORT: str = os.environ["TENSORBOARD_PORT"]
FLASK_PORT: str = os.environ["FLASK_PORT"]


@tensorboard_app.route("/")
@tensorboard_app.route("/index")
def index() -> str:
    return (
        "If you want to launch tensorboard and get the appropriate URL, "
        " try writing the url as follows: "
        f"'{SERVER_PROTOCOL}://{WAN_HOST_IP}:{FLASK_PORT}"
        "/launch_tensorboard/{rel_path}'"
    )


def _tensorboard_main(logdir: pathlib.PosixPath) -> None:
    tb = tensorboard.program.TensorBoard()
    tb.configure(
        argv=[
            None,
            "--logdir", logdir.as_posix(),
            "--port", TENSORBOARD_PORT,
            "--host", LAN_HOST_IP
        ]
    )
    tb.main()


@tensorboard_app.route("/launch_tensorboard/<path:rel_path>")
def render_tansorboard(rel_path: str) -> None:
    tensorboard_log_dir: pathlib.PosixPath = PREFIX_PATH / rel_path
    if not tensorboard_log_dir.exists() \
            or not tensorboard_log_dir.is_dir() \
            or sum(
                1 for _ in tensorboard_log_dir.glob("events.out.tfevents.*")
            ) == 0:
        return (
            "HTTP 404 – Not Found\n"
            "The requested resource could not be located on this server."
            " Please ensure the specified log directory is correct, and try again."
        )
    else:
        tb_process = Process(
            target=_tensorboard_main,
            args=(tensorboard_log_dir,)
        )
        tb_process.start()
        if not hasattr(render_tansorboard, "prev_tb_process_pid"):
            render_tansorboard.prev_tb_process_pid = None
        if render_tansorboard.prev_tb_process_pid:
            os.kill(render_tansorboard.prev_tb_process_pid, signal.SIGTERM)
            render_tansorboard.prev_tb_process_pid = None
        render_tansorboard.prev_tb_process_pid = tb_process.pid
        #  TODO: find out is 'tb_process.join()' needed indeed
        return (
            f"Tensorboard for `{rel_path}` has been launched, "
            "you can reach it through the link: "
            f"{SERVER_PROTOCOL}://{WAN_HOST_IP}:{TENSORBOARD_PORT}"
            "/#scalars&_smoothingWeight=0"
        )
    
