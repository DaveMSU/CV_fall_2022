import functools
import logging
import random
import typing as tp  # TODO: to better the typing in this file

import numpy as np
import torch
from scipy.spatial.distance import cosine

_BASE_TEMPLATE = "{0}: {1}(`{2}`)"
_IGNORED_TEMPLATE = "{0}: ignored object of type `{1}`"
_OBJ_WITH_SHAPE_TEMPLATE = ("{0}: array-like of type `{1}` and shape `{2}`")
_NDARRAY_OR_TORCH_TENSOR_TEMPLATE = (
    "{0}: array-like of type `{1}`, shape `{2}`, dtype `{3}`"
    " and mean cossine for axis=0 `{4}`"
)
AMOUNT_OF_PAIRS_TO_CONSIDER = 100


# TODO: answer to myself if it's really needed?
def _calculate_mean_cossine_for_axis_0(
        arr: tp.Union[np.ndarray, torch.Tensor]
):
    assert arr.ndim >= 1, arr
    if arr.shape[0] == 1:
        return 1.0
    else:
        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().data.numpy()
        row_ids: tp.Set[tp.Tuple[int, int]] = set()
        cum_mean: float = 0
        while len(row_ids) < min(
                AMOUNT_OF_PAIRS_TO_CONSIDER,
                arr.shape[0] * (arr.shape[0] - 1) // 2
        ):
            first_row: int = random.randint(0, arr.shape[0] - 2)
            second_row: int = random.randint(first_row + 1, arr.shape[0] - 1)
            if (first_row, second_row) not in row_ids:
                row_ids.add((first_row, second_row))
                cum_mean += (
                    cosine(
                        arr[first_row].reshape(-1),
                        arr[second_row].reshape(-1)
                    ) - cum_mean
                ) / len(row_ids)
        return cum_mean


# TODO: fix ignore_args usage, because there case where it's easy to hack
def wrap_in_logger(level: str, ignore_args: tp.Tuple[int, ...] = ()):
    def decorator(func: tp.Callable) -> tp.Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            specific_property_mention: str = ""
            for _sp in ["classmethod", "property", "staticmethod"]:
                if isinstance(func, __builtins__[_sp]):
                    # TODO: check why above cond doesn't detect any of them
                    specific_property_mention = f" ({_sp})"
                    break
            log = getattr(
                logging.getLogger(
                    f"{func.__qualname__}{specific_property_mention}"
                ),
                level
            )

            log("the function is going to be invoked with ...")
            for name, iter_obj in [
                    ("*args", enumerate(args)),
                    ("**kwargs", kwargs.items())
            ]:
                log(f"{name}:")
                for _id, obj in iter_obj:
                    if (type(_id) is int) and (_id in ignore_args):
                        log(_IGNORED_TEMPLATE.format(_id, type(obj)))
                    elif hasattr(obj, "shape") and sum(obj.shape) > 5:
                        if isinstance(obj, (np.ndarray, torch.Tensor)):
                            log(
                                _NDARRAY_OR_TORCH_TENSOR_TEMPLATE.format(
                                    _id,
                                    type(obj),
                                    obj.shape,
                                    obj.dtype,
                                    _calculate_mean_cossine_for_axis_0(obj)
                                )
                            )
                        else:
                            log(
                                _OBJ_WITH_SHAPE_TEMPLATE.format(
                                    _id, type(obj), obj.shape
                                )
                            )
                    else:
                        log(_BASE_TEMPLATE.format(_id, obj, type(obj)))
            result: tp.Any = func(*args, **kwargs)
            log(
                "the function has finished and returned:"
                f" `{result}`(`{type(result)}`)"
            )
            return result
        return wrapper
    return decorator
