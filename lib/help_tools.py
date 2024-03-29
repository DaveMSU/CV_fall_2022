import argparse
import logging
import sys
import typing as tp

import numpy as np


class RunningMeansHandler:
    def __init__(self):
        self._value: tp.Optional[tp.Union[float, np.ndarray]] = None
        self._counter: int = 0

    def add(
            self,
            value: tp.Optional[tp.Union[float, np.ndarray]],
            n: int = 1
    ) -> None:
        assert n >= 0
        assert (self._value is None) or (type(self._value) is type(value))
        if value is None:
            if self._counter != 0:
                raise ValueError(
                    "{self._counter=}, but have to be 0, if value is None!"
                )
        elif self._value is None:
            self._counter, self._value = self._counter + n, value
            assert self._counter == n
        elif self._value is not None:
            self._counter += n
            self._value += (value - self._value) * (n / self._counter)
        else:
            assert False, "Unreachable line!"

    def get_value(self) -> tp.Union[float, np.ndarray]:
        return self._value


def parse_args() -> tp.Dict[str, tp.Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', required=True)
    parser.add_argument('--learn', required=True)
    return parser.parse_args()


def make_logger(
        logger_name: str,
        logging_file: tp.Union[str, bool],
        stdout: bool
    ) -> logging.Logger:
    """
    :param logger_name: just name of logger.
    :param logging_file: name of logging file,
    often ended with suffix '.log'. If false
    occured, then it is don't using FileHandler.
    :param stdout: Do add logging in stdout stream or don't.

    :return: a logger into which data could be written.
    """
    logger: logging.Logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handlers: tp.List[logging.StreamHandler] = []
    if isinstance(logging_file, str):
        handlers.append(logging.FileHandler(logging_file))
    elif isinstance(logging_file, bool) and (logging_file == True):
        raise ValueError("'logging_file' variable shouldn't be True!")

    if stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    if not len(handlers):
        raise Exception(
            "There are no logging handlers, but"
            " there must be at least one handler."
        )
    for h in handlers:
        h.setLevel(logging.DEBUG)
        h.setFormatter(formatter)
        logger.addHandler(h)
    return logger

