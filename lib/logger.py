import functools
import logging
import typing as tp  # TODO: to better the typing in this file

_BASE_TEMPLATE = "{0}: {1}(`{2}`)"
_IGNORED_TEMPLATE = "{0}: ignored object of type `{1}`"
_OBJ_WITH_SHAPE_TEMPLATE = ("{0}: array-like of type `{1}` and shape `{2}`")
_OBJ_WITH_SHAPE_AND_DTYPE_TEMPLATE = (
    "{0}: array-like obj of type `{1}`, shape `{2}` and dtype `{3}`"
)


# TODO: fix ignore_args usage, because there case where it's easy to hack
def wrap_in_logger(level: str, ignore_args: tp.Tuple[int, ...] = ()):
    def decorator(func: tp.Callable) -> tp.Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            specific_property_mention: str = ""
            for _sp in ["classmethod", "property", "staticmethod"]:
                if isinstance(func, __builtins__[_sp]):  # TODO: check why it doesn't detect any of them
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
                        if hasattr(obj, "dtype"):
                            log(
                                _OBJ_WITH_SHAPE_AND_DTYPE_TEMPLATE.format(
                                    _id, type(obj), obj.shape, obj.dtype
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
            log(f"the function has been finished and returned: `{result}`")
            return result
        return wrapper
    return decorator
