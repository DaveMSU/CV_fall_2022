import argparse
import logging
import sys
import typing as tp


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

