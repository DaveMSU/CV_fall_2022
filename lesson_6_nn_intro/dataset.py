import pathlib
import typing as tp

import cv2
import torch
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ImagePointsDataset(torch.utils.data.Dataset):
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
                "target": points
            }
            for file_name, points in train_gt.items()
        ]
        
        for pair in self._items:
            assert pair["path"].exists(), f"File '{pair['path']}' do not exists!"
            
        train_size = round(len(self._items) * train_fraction)
        
        if mode == "train":
            self._items = self._items[:train_size]
        elif mode == "val":
            self._items = self._items[train_size:]
        else:
            assert f"Mode '{mode}' undefined!"
            
        self._X_size = new_size[0]
        self._Y_size = new_size[1]
        
    
    @property
    def shape(self):
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

        ## read image 
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.
        
        # Change shapes
        orig_shape = image.shape
        self._items[index]["shape"] = orig_shape
        image = cv2.resize(image, (self._X_size, self._Y_size))
        target[::2] = target[::2] / orig_shape[0]
        target[1::2] = target[1::2] / orig_shape[1]
        
        # to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))
        target = torch.from_numpy(target.astype(np.float32))
        
        return image, target, torch.Tensor(orig_shape)


def convert_targets_shape(
        target: np.ndarray,
        shape: tp.Union[tp.Tuple[int, int], np.ndarray]
) -> np.ndarray:
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
):
    img_arr, target, shape = ds[index]
    img_arr = img_arr.numpy().transpose(1, 2, 0)
    target = convert_targets_shape(target, ds.shape)
    target = target.numpy()
    xs = target[::2]
    ys = target[1::2]
    plt.imshow(img_arr)
    plt.scatter(xs, ys)

