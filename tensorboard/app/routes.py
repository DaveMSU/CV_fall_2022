import os
import signal

import flask
import requests
import tensorboard
from multiprocessing import Process
from pathlib import Path

from . import tensorboard_app


COURSE_DIR_PATH: Path = Path("/var/lib/storage/resources/cv_fall_2022/")
SERVER_PROTOCOL: str = "http"  # TODO: try to move app to https
LOCAL_HOST_IP: str = "192.168.0.239"
GLOBAL_HOST_IP: str = "77.242.105.252"
TENSORBOARD_PORT: str = "8443"
FLASK_PORT: str = "5001"


@tensorboard_app.route("/")
@tensorboard_app.route("/index")
def index() -> str:
    return (
        "If you want to launch tensorboard and get the appropriate URL, "
        " try writing the url as follows: "
        f"'{SERVER_PROTOCOL}://{GLOBAL_HOST_IP}:{FLASK_PORT}"
        "/launch_tensorboard/{lesson_dir}/{nn_dir}'"
    )


def _tensorboard_main(lesson_dir: str, nn_dir: str) -> None:
    tb = tensorboard.program.TensorBoard()
    tb.configure(
        argv=[
            None,
            "--logdir", f"/var/lib/storage/resources/cv_fall_2022/{lesson_dir}/experiments/{nn_dir}/tensorboard_runs",
            "--port", TENSORBOARD_PORT,
            "--host", LOCAL_HOST_IP
        ]
    )
    tb.main()


@tensorboard_app.route("/launch_tensorboard/<lesson_dir>/<nn_dir>")
def render_tansorboard(lesson_dir: str, nn_dir: str) -> None:
    lesson_dir_abs_path: Path = COURSE_DIR_PATH / lesson_dir
    nn_dir_abs_path: Path = lesson_dir_abs_path / "experiments" / nn_dir
    if not os.path.isdir(lesson_dir_abs_path):
        return f"Sorry, dir `{lesson_dir_abs_path}` does not exist."
    elif not os.path.isdir(nn_dir_abs_path):
        return f"Sorry, dir `{nn_dir_abs_path}` doest not exist."
    elif not os.path.isdir(nn_dir_abs_path / "tensorboard_runs"):
        return (
            "Unfortunately there is no data for visualization."
            f" Dir `{nn_dir_abs_path}` exists,"
            f" but `{nn_dir_abs_path}/tensorboard_runs` does not."
        )

    tb_process = Process(
        target=_tensorboard_main,
        args=(lesson_dir, nn_dir)
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
        f"Tensorboard for `{lesson_dir}/{nn_dir}` has been launched, "
        "you can reach it through the link: "
        f"{SERVER_PROTOCOL}://{GLOBAL_HOST_IP}:{TENSORBOARD_PORT}"
        "/#scalars&_smoothingWeight=0"
    )

