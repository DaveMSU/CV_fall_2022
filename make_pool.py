import argparse
import json
import pathlib
import pickle
import typing as tp

import cv2
import numpy as np

import lib.datasets
from lib.transforms import (
    FacePointsRandomCropTransform,
    FacePointsRandomHorizontalFlipTransform
)


def _read_csv(filename: pathlib.Path) -> tp.Dict[str, np.ndarray]:
    res = {}
    with open(filename) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([float(x) for x in parts[1:]], dtype='float64')
            res[parts[0]] = coords
    return res


def main():
    # make pool configuration file loading.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = pathlib.Path(args.config)
    
    with open(config_path) as f:
        config = json.load(f)
    
    # train/val data preparing.
    train_img_dir = pathlib.Path(config["train_data_dir"])
    train_gt: tp.Dict[str, np.ndarray] = _read_csv(config["labels"])
    dataset_params = config["dataset_params"]
    train_fraction = dataset_params["train_fraction"]
    if dataset_params["augmentation"] is None:
        transforms = None
    else:
        transforms = [
            globals()[dict_["transform_type"]](**dict_["params"])
                for dict_ in dataset_params["augmentation"]
        ]

    datasets = {
        mode: getattr(lib.datasets, config["dataset_type"])(
            mode=mode,
            train_fraction=train_fraction,
            data_dir=train_img_dir,
            train_gt=train_gt,
            new_size=dataset_params["new_size"],
            transforms=transforms
        ) for mode in ["train", "val"]
    }

    # dumping dataset somewhere.
    with open(config["train_dump_path"], 'wb') as f:
        pickle.dump(datasets["train"], f)
    
    with open(config["val_dump_path"], 'wb') as f:
        pickle.dump(datasets["val"], f)


if __name__ == "__main__":
    main()

