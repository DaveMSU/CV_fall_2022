import typing as tp

import torch
import numpy as np
from matplotlib import pyplot as plt


def convert_targets_shape(
        target: np.ndarray,
        shape: tp.Union[tp.Tuple[int, int], np.ndarray]
) -> np.ndarray:
    target = target.copy()
    if isinstance(shape, tuple):
        target[::2] = target[::2] * shape[0]
        target[1::2] = target[1::2] * shape[1]
    else:
        target[:,::2] = target[:,::2] * shape[:,0].reshape(-1, 1)
        target[:,1::2] = target[:,1::2] * shape[:,1].reshape(-1, 1)
    return target


def show_face(
        ds: torch.utils.data.Dataset,
        index: int
) -> None:
    img_arr, target, shape = ds[index]
    img_arr = img_arr.numpy().transpose(1, 2, 0)
    target = target.numpy()
    target = convert_targets_shape(target, ds.shape)
    xs = target[::2]
    ys = target[1::2]
    plt.imshow(img_arr)
    plt.scatter(xs, ys)

