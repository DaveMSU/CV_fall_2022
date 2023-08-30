import typing as tp

import cv2
import numpy as np
import pathlib
import torch
from PIL import Image

from .base_dataset import BaseImageDataset


class ImageClassifyDataset(BaseImageDataset):
    def __init__(self, classes_num: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._classes_num = classes_num

    @property
    def classes_num(self):
        return self._classes_num

    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self._items[index]
        img_path = pair["path"]
        target = np.array(pair["target"])  # np.array(int)

        # Load image.
        image = Image.open(img_path).convert('RGB')
        if self._transforms:
            for transform in self._transforms:
                image = transform(image)
        image = np.array(image).astype(np.float32) / 255.

        # Change shapes.
        orig_shape = image.shape
        self._items[index]["shape"] = orig_shape
        new_X_size = self._X_size if self._X_size else orig_shape[0]
        new_Y_size = self._Y_size if self._Y_size else orig_shape[1]
        image = cv2.resize(image, (new_X_size, new_Y_size))

        # Convert to tensor.
        image = torch.from_numpy(image.transpose(2, 0, 1))
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        # Do one hot encoding of the target.
        blank_line = np.zeros(self._classes_num, dtype=np.float32)
        one_hot_line = blank_line.copy()
        one_hot_line[target] = 1.0
        target = torch.from_numpy(one_hot_line)
        return image, target, torch.Tensor(orig_shape)

    @staticmethod
    def read_markup(file_path: pathlib.Path) -> tp.Dict[str, np.ndarray]:
        path_to_values: tp.Dict[str, np.ndarray] = dict()
        with open(file_path) as fhandle:
            next(fhandle)  # skip the line with the column names.
            for line in fhandle:
                file_name, class_id = line.rstrip('\n').split(',')
                path_to_values[file_name] = int(class_id)
        return path_to_values

