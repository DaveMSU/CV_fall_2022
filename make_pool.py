import argparse
import json
import pathlib
import pickle

import lib.datasets
import lib.transforms


def main():
    # make pool configuration file loading.
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config_path = pathlib.Path(args.config)

    with open(config_path) as f:
        config = json.load(f)

    # train/val data preparing.
    for pool_config in config["pools"]:
        if pool_config["dataset_params"]["transforms"] is None:
            transforms = None
        else:
            transforms = [
                getattr(lib.transforms, dict_["transform_type"])(**dict_["params"])
                for dict_ in pool_config["dataset_params"]["transforms"]
            ]
        DatasetClass = getattr(lib.datasets, pool_config["dataset_type"])
        dataset_params = dict(**pool_config["dataset_params"])
        dataset_params["inclusion_condition"] = \
            eval(pool_config["dataset_params"]["inclusion_condition"])
        dataset_params["data_dir"] = pathlib.Path(config["train_data_dir"])
        dataset_params["markup"] = DatasetClass.read_markup(config["labels"])
        dataset_params["transforms"] = transforms
        dataset = DatasetClass(**dataset_params)

        # dumping dataset somewhere.
        with open(pool_config["dump_path"], 'wb') as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    main()

