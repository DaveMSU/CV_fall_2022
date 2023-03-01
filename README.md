# Code for making pools for trainling/validation/etc - make_pool.py
### Example of running:
```
python make_pool.py --config=./lesson_X/configs/pool_making/base.json
```

# Code for training networks, all output you can find at storage/resources - train.py
### Example of running:
```
python train.py --net=./lesson_X/configs/architectures/simple_cnn.json --learn=./lesson_X/configs/learning_process/config.json
```

# For all raw data like images (png/jpg/...) or labels (csv/txt/...) look into:
```
/var/lib/storage/data
```

# For output data like pools, logs, model dumps, etc look into:
```
/var/lib/storage/resources
```

