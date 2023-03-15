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
    for pool_config in config["pools"]:
        if pool_config["dataset_params"]["augmentation"] is None:
            transforms = None
        else:
            transforms = [
                globals()[dict_["transform_type"]](**dict_["params"])
                    for dict_ in pool_config["dataset_params"]["augmentation"]
            ]

        dataset = getattr(lib.datasets, pool_config["dataset_type"])(
            mode=pool_config["mode"],
            train_fraction=pool_config["dataset_params"]["train_fraction"],
            data_dir=pathlib.Path(config["train_data_dir"]),
            train_gt=_read_csv(config["labels"]),
            new_size=pool_config["dataset_params"]["new_size"],
            transforms=transforms
        )

        # dumping dataset somewhere.
        with open(pool_config["dump_path"], 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    main()

