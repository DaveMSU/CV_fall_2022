import copy
import pathlib
import typing as tp

import cv2
import numpy as np
import torch
from PIL import Image

from .base_dataset import BaseImageDataset


class ImagePointsDataset(BaseImageDataset):
    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self._items[index]
        img_path = pair["path"]
        target = copy.deepcopy(pair["target"])

        # Load image.
        image = Image.open(img_path).convert('RGB')
        if self._transforms:
            for transform in self._transforms:
                image, target = transform(image, target)
        image = np.array(image).astype(np.float32) / 255.

        # Change shapes.
        orig_shape = image.shape
        self._items[index]["shape"] = orig_shape
        image = cv2.resize(image, (self._X_size, self._Y_size))
        target[::2] = target[::2] / orig_shape[0]
        target[1::2] = target[1::2] / orig_shape[1]

        # Convert to tensor.
        image = torch.from_numpy(image.transpose(2, 0, 1))
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
        target = torch.from_numpy(target.astype(np.float32))
        return image, target, torch.Tensor(orig_shape)

    @staticmethod
    def read_markup(file_path: pathlib.Path) -> tp.Dict[str, np.ndarray]:
        path_to_values: tp.Dict[str, np.ndarray] = dict()
        with open(file_path) as fhandle:
            next(fhandle)  # skip the line with the column names.
            for line in fhandle:
                file_name, *raw_coords = line.rstrip('\n').split(',')
                coords = np.array(
                    [float(x) for x in raw_coords],
                    dtype='float64'
                )
                path_to_values[file_name] = coords
        return path_to_values
