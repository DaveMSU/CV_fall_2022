import abc
import copy
import typing as tp

import numpy as np
import pathlib
import torch


class BaseImageDataset(torch.utils.data.Dataset, abc.ABC):
    def __init__(
            self,
            inclusion_condition: tp.Callable[[str], bool],
            data_dir: pathlib.PosixPath,
            markup: tp.Dict[str, tp.Union[np.ndarray, int]],
            new_size: tp.Tuple[int, int],
            transforms: tp.Union[
                tp.Optional[tp.List[tp.Callable]],
                torch.nn.Sequential
            ] = None
    ):
        self._items = [
            {
                "path": data_dir/file_name,
                "target": copy.deepcopy(target)
            }
            for file_name, target in markup.items()
            if inclusion_condition(file_name)
        ]
        np.random.seed(7)
        permuted_indexes = np.random.permutation(np.arange(len(self._items)))
        self._items = [self._items[i] for i in permuted_indexes]

        assert data_dir.exists(), f"Dir '{data_dir}' do not exists!"
        for pair in self._items:
            assert pair["path"].exists(),\
                f"File '{pair['path']}' does not exist!"

        self._transforms = transforms
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

    @abc.abstractmethod
    def __getitem__(
            self,
            index: int
    ) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_markup(
            file_path: pathlib.Path
    ) -> tp.Dict[str, tp.Union[np.ndarray, int]]:
        raise NotImplementedError
