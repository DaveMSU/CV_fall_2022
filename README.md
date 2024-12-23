# dl - is the only entrance for all manipulation with deep learning processes.

### weld_dataset: creats welded torch.Tensor in binary format (hdf5) that is ready for DataLoaders, example:
```
./dl weld_dataset --config ./exps/lesson_6_nn_intro/configs/dataset_welding.json
```

### train: starts learning process, example:
```
./dl train --architecture_config ./exps/lesson_6_nn_intro/configs/architecture.json --learning_config ./exps/lesson_6_nn_intro/configs/learning_process.json --log-level INFO
```

### tensorboard: creates flask server with tensorboard monitoring on it, the only way to launch it:
```
./dl tensorboard
```

# For all raw data like images (png/jpg/...) or labels (csv/txt/...) and welded dataset look into:
```
/var/lib/storage/data
```

# For output data like logs, model dumps, etc look into:
```
/var/lib/storage/resources
```

