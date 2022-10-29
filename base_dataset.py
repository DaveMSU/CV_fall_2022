import pathlib
import typing as tp

import torch
import numpy as np
from PIL import Image


class BaseImageDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode: str,
            train_fraction: float,
            data_dir: pathlib.PosixPath,
            train_gt: tp.Dict[str, np.ndarray],
            new_size: tp.Tuple[int, int] = (64, 64),
            transforms: tp.Union[
                tp.Optional[tp.List[tp.Callable],
                torch.nn.Sequential
            ] = None
    ):
        self._items = [
            {
                "path": data_dir/file_name,
                "target": target
            }
            for file_name, target in train_gt.items()
        ]
        np.random.seed(7)
        self._items = np.random.permutation(self._items)
        
        assert data_dir.exists(), f"Dir '{data_dir}' do not exists!" 
        for pair in self._items:
            assert pair["path"].exists(), f"File '{pair['path']}' do not exists!"
            
        self._transforms = transforms

        train_size = round(len(self._items) * train_fraction)
        self._mode = mode        
        if self._mode == "train":
            self._items = self._items[:train_size]
        elif self._mode == "val":
            self._items = self._items[train_size:]
        elif self._mode == "test":
            pass
        else:
            assert f"Mode '{self._mode}' undefined!"
            
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
        raise NotImplemented

