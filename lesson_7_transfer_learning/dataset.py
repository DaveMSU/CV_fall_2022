import pathlib
import typing as tp

import cv2
import torch
import numpy as np
from PIL import Image


class ImageClassifyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode: str,
            train_fraction: float,
            data_dir: pathlib.PosixPath,
            train_gt: tp.Dict[str, np.ndarray],
            new_size: tp.Tuple[int, int] = (64, 64)
    ):
        assert data_dir.exists(), f"Dir '{data_dir}' do not exists!"
        
        self._items = [
            {
                "path": data_dir/file_name,
                "target": class_
            }
            for file_name, class_ in train_gt.items()
        ]

        np.random.seed(7)
        self._items = np.random.permutation(self._items)
        
        for pair in self._items:
            assert pair["path"].exists(), f"File '{pair['path']}' do not exists!"
            
        train_size = round(len(self._items) * train_fraction)

        self._mode = mode        
        if mode == "train":
            self._items = self._items[:train_size]
        elif mode == "val":
            self._items = self._items[train_size:]
        elif mode == "test":
            pass
        else:
            assert f"Mode '{mode}' undefined!"
            
        self._X_size = new_size[0]
        self._Y_size = new_size[1]

    
    @property
    def paths(self) -> tp.List[str]:
        return [pair["path"] for pair in self._items]
        
    
    @property
    def shape(self) -> tp.Tuple[int, int]:
        return (self._X_size, self._Y_size)
    
    
    def __len__(self):
        return len(self._items)
    
    
    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pair = self._items[index]
        img_path = pair["path"]
        target = pair["target"].copy()

        ## Load image.
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.
        
        # Change shapes.
        orig_shape = image.shape
        self._items[index]["shape"] = orig_shape
        image = cv2.resize(image, (self._X_size, self._Y_size))
        
        # Convert to tensor.
        image = torch.from_numpy(image.transpose(2, 0, 1))
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)        
        target = torch.from_numpy(target.astype(np.float32))
        
        return image, target, torch.Tensor(orig_shape)

