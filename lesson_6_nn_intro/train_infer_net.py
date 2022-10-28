import pathlib
import typing as tp

import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader


def train_detector(
        train_gt: tp.Dict[str, np.ndarray],
        train_img_dir: str,
        fast_train: bool = False,
        net_arch: tp.Dict[str, tp.Any] = NET_ARCHITECTURE,
        learning_process: tp.Dict[str, tp.Any] = LEARNING_PROCESS
) -> NeuralNetwork:

    # Create dataloaders.
    train_img_dir = pathlib.Path(train_img_dir)
    dataset_params = learning_process["dataset_params"]
    train_fraction = dataset_params["train_fraction"]
    train_dataset = ImagePointsDataset(
        mode="train",
        train_fraction=train_fraction,
        data_dir=train_img_dir,
        train_gt=train_gt,
        new_size=dataset_params["new_size"]
    )
    val_dataset = ImagePointsDataset(
        mode="val",
        train_fraction=train_fraction,
        data_dir=train_img_dir,
        train_gt=train_gt,
        new_size=dataset_params["new_size"]
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size = dataset_params["train_batch_size"],
        shuffle=False
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size = dataset_params["val_batch_size"],
        shuffle=False
    )    

    # Init loss, optimizer, network.
    loss_params = learning_process["hyper_params"]["loss"]
    optimizer_params = learning_process["hyper_params"]["optimizer"]
    epoch_nums = learning_process["hyper_params"]["epoch_nums"]

    net = NeuralNetwork(net_arch)
    loss = getattr(torch.nn, loss_params["loss_type"])(**loss_params["params"])
    optimizer = getattr(torch.optim, optimizer_params["optimizer_type"])(
        net.parameters(),
        **optimizer_params["params"]
    )
    optimizer.zero_grad()

    best_val_loss = float("inf")

    # Train and validate network.
    for epoch in range(1 if fast_train else epoch_nums):
        print(f"{epoch=}:")

        loss_history = []
        for X, y, _ in train_dataloader:
            optimizer.zero_grad()
            y_pred = net(X)
            loss_value = loss(y_pred, y)
            loss_value.backward()
            optimizer.step()
            cur_train_loss = loss_value.cpu().data.item()
            loss_history.append(cur_train_loss)
            print("train_loss: %.4f" % cur_train_loss, end='')

        train_loss = np.mean(loss_history)
        print("train_loss:	%.5f" % train_loss)

        loss_history = []
        for X, y, shape in val_dataloader:
            shape = shape.numpy()
            y_pred = net(X)
            loss_value = loss(y_pred, y) 
            loss_history.append(loss_value.cpu().data.item())

        val_loss = np.mean(loss_history)
        print("val_loss:	%.5f" % val_loss)
        print("val_quality:	%.5f" % (val_loss * 100**2))

        if not fast_train and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    'model_state_dict': net.state_dict()
                },
                pathlib.Path("./facepoints_model.ckpt")
            )
            print("*")        
        print()

    # Get best one network.
    if not fast_train:
        net = NeuralNetwork(net_arch)
        checkpoint = torch.load("./facepoints_model.ckpt")
        net.load_state_dict(checkpoint['model_state_dict'])

    return net


def detect(
        model_filename: str,
        test_img_dir: str,
        net_arch: tp.Dict[str, tp.Any] = NET_ARCHITECTURE,
        learning_process: tp.Dict[str, tp.Any] = LEARNING_PROCESS
) -> tp.Dict[str, np.ndarray]:

    # Load network.
    net = NeuralNetwork(net_arch)
    checkpoint = torch.load("./facepoints_model.ckpt")
    net.load_state_dict(checkpoint['model_state_dict'])

    # Create dataloader.
    test_img_dir = pathlib.Path(test_img_dir)
    test_gt = {path.name: np.array([]) for path in test_img_dir.iterdir()}

    dataset_params = learning_process["dataset_params"]
    test_dataset = ImagePointsDataset(
        mode="test",
        train_fraction=0.0,
        data_dir=test_img_dir,
        train_gt=test_gt,
        new_size=dataset_params["new_size"]
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size = dataset_params["val_batch_size"],
        shuffle=False
    )

    # Infer network on test images.
    y_pred_history = []
    for i, (X, y, shape) in enumerate(test_dataloader):
        print(f"iter={i}/{len(test_dataloader) - 1}", end='')
        y_pred = net(X)
        y_pred_reshaped = convert_targets_shape(
            y_pred.cpu().data.numpy().copy(),
            shape.cpu().data.numpy().copy()
        )
        y_pred_history.extend(y_pred_reshaped.tolist())

    # Create returned dict.
    for path, points in zip(test_dataset.paths, y_pred_history):
        test_gt[path.name] = np.array(points)

    return test_gt

