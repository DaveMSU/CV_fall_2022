import sys; sys.path.append('../')

from collections import defaultdict
from itertools import chain
import typing as tp

import cv2
import torch
import numpy as np
import pathlib
from PIL import Image
from matplotlib import pyplot as plt

from base_dataset import BaseImageDataset


class ImageClassifyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            mode: str,
            val_type: str,
            train_fraction: float,
            root_folder: pathlib.PosixPath,
            classes_num: int,
            new_size: tp.Tuple[int, int] = (64, 64),
            transforms: tp.Union[
                tp.Optional[tp.List[tp.Callable]],
                torch.nn.Sequential
            ] = None
    ):
        assert val_type in ["random", "stratify"], "Something wrong!"
        assert mode in ["train", "val", "test"], "Something wrong!"

        self._class_name_to_id = {  # path to class_name
            sub_folder.name: indx
                for indx, sub_folder in enumerate(root_folder.iterdir())
        }

        self._items = [
            {
                "path": full_path,
                "target": self._class_name_to_id[sub_folder.name]
            }
                for sub_folder in root_folder.iterdir()
                for full_path in sub_folder.iterdir()
        ]

        np.random.seed(7)
        self._items = np.random.permutation(self._items)

        self._items_groups = defaultdict(list)
        for indx, item in enumerate(self._items):
            class_name = item["path"].parent.name
            self._items_groups[class_name].append(indx)
#        assert len(self._items_groups) == classes_num, len(self._items_groups)

        ts = round(len(self._items) * train_fraction)  # train_size
        vs = len(self._items) - ts  # val_size
        self._mode = mode
        self._val_type = val_type
        if self._val_type == "random":
            if self._mode == "train":
                self._items = self._items[:ts]
#                self._items_groups = {
#                    class_name: [i for i in indexes if i < ts]
#                        for class_name, indexes in self._items_groups.items()
#                }
            elif self._mode == "val":
                self._items = self._items[ts:]
#                self._items_groups = {
#                    class_name: [i - ts for i in indexes if i - ts >= 0]
#                        for class_name, indexes in self._items_groups.items()
#                }
        else:  # val_type == "stratify"
            train_uncomplete_classes = 0
            val_uncomplete_classes = 0
            items_groups = dict()
            for class_name, indexes in self._items_groups.items():
                p = len(indexes) / len(self._items)
                val_elems_amount = round(vs * p)
                assert len(indexes) >= val_elems_amount >= 0
                if val_elems_amount == len(indexes):
                    train_uncomplete_classes += 1
                    if self._mode == "val":
                        items_groups[class_name] = indexes[:]
                elif 0 < val_elems_amount < len(indexes):
                    if self._mode == "train":
                        items_groups[class_name] = indexes[val_elems_amount:]
                    else:
                        items_groups[class_name] = indexes[:val_elems_amount]
                else:
                    val_uncomplete_classes += 1
                    if self._mode == "train":
                        items_groups[class_name] = indexes[:]
            self._items = [
                self._items[i] for i in chain.from_iterable(items_groups.values())
            ]
            self._train_uncomplete_classes = train_uncomplete_classes
            self._val_uncomplete_classes = val_uncomplete_classes
        
        self._items_groups = defaultdict(list)
        for indx, item in enumerate(self._items):
            class_name = item["path"].parent.name
            self._items_groups[class_name].append(indx)
        
        self._transforms = transforms
        self._classes_num = classes_num

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
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        pair = self._items[index]
        img_path = pair["path"]
        target = np.array(pair["target"])  # np.array(int)

        ## Load image.
        image = Image.open(img_path).convert('RGB')
        if self._transforms:
            for transform in self._transforms:
                image, target = transform(image, target)
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
        if self._mode != "test":
            blank_line[int(target)] = 1.0
        target = torch.from_numpy(blank_line.copy())
        
        return image, target

