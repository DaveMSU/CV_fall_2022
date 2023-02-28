import typing as tp

import cv2
import torch
import numpy as np
from PIL import Image

from ..base_dataset import BaseImageDataset


class ImagePointsDataset(BaseImageDataset): 
    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self._items[index]
        img_path = pair["path"]
        target = pair["target"].copy()

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
        if self._mode != "test":
            target[::2] = target[::2] / orig_shape[0]
            target[1::2] = target[1::2] / orig_shape[1]

        # Convert to tensor.
        image = torch.from_numpy(image.transpose(2, 0, 1))
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)        
        target = torch.from_numpy(target.astype(np.float32))
        
        return image, target, torch.Tensor(orig_shape)

