import json
import pathlib
import typing as tp

import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


NET_ARCHITECTURE = [
    {
        "layer_type": "Conv2d",
        "params": {
            "in_channels": 3,
            "out_channels": 64,
            "kernel_size": [3, 3],
            "stride": 1,
            "padding": 0
        }
    },
    {
        "layer_type": "ReLU",
        "params": {}
    },
    {
        "layer_type": "MaxPool2d",
        "params": {
            "kernel_size": [2, 2],
            "stride": 2,
            "padding": 0
        }
    },
    {
        "layer_type": "Conv2d",
        "params": {
            "in_channels": 64,
            "out_channels": 128,
            "kernel_size": [3, 3],
            "stride": 1,
            "padding": 0
        }        
    },
    {
        "layer_type": "ReLU",
        "params": {}
    },    
    {
        "layer_type": "MaxPool2d",
        "params": {
            "kernel_size": [2, 2],
            "stride": 2,
            "padding": 0
        }        
    },
    {
        "layer_type": "Conv2d",
        "params": {
            "in_channels": 128,
            "out_channels": 256,
            "kernel_size": [3, 3],
            "stride": 1,
            "padding": 0
        }           
    },
    {
        "layer_type": "ReLU",
        "params": {}
    },    
    {
        "layer_type": "MaxPool2d",
        "params": {
            "kernel_size": [2, 2],
            "stride": 2,
            "padding": 0
        }                
    },
    {
        "layer_type": "Flatten",
        "params": {}
    },
    {
        "layer_type": "Linear",
        "params": {
            "in_features": 9216,
            "out_features": 64            
        }
    },
    {
        "layer_type": "ReLU",
        "params": {}
    },   
    {
        "layer_type": "Linear",
        "params": {
            "in_features": 64,
            "out_features": 28            
        }
    },  
    {
        "layer_type": "Sigmoid",
        "params": {}
    }
]


LEARNING_PROCESS = {
    "dataset_params": {
        "train_fraction": 0.9,
        "new_size": [64, 64],
        "train_batch_size": 50,
        "val_batch_size": 100
    },
    "hyper_params": {
        "loss": {
            "loss_type": "MSELoss",
            "params": {}
        },
        "optimizer": {
            "optimizer_type": "SGD",
            "params": {
                "lr": 0.1
            }
        },
        "epoch_nums": 30
    }
}


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
        ][:1000]

        np.random.seed(7)
        self._items = np.random.permutation(self._items)
        
        for pair in self._items:
            assert pair["path"].exists(), f"File '{pair['path']}' do not exists!"
            
        train_size = round(len(self._items) * train_fraction)

        self._mode = mode        
        if mode == "train":
            self._items = self._items[:train_size]
        elif mode == "val":
            self._items = self._items[train_size:]
        elif mode == "test":
            pass
        else:
            assert f"Mode '{mode}' undefined!"
            
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
        if self._mode != "test":
            target[::2] = target[::2] / orig_shape[0]
            target[1::2] = target[1::2] / orig_shape[1]
        
        # to tensor
        image = torch.from_numpy(image.transpose(2, 0, 1))
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)
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


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)
    

class NeuralNetwork(torch.nn.Module):  
    def __init__(self, architecture: tp.List[tp.Dict[str, tp.Any]]):
        super().__init__()
        self._layers_seq = torch.nn.Sequential(
            *[
                getattr(torch.nn, layer["layer_type"])(**layer["params"]) 
                    if hasattr(torch.nn, layer["layer_type"]) 
                    else globals()["Flatten"](**layer["params"])
                for layer in architecture
            ]
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers_seq(x)


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
            print("train_loss: %.4f" % cur_train_loss, end='\r')

        train_loss = np.mean(loss_history)
        print("train_loss:\t%.5f" % train_loss)

        loss_history = []
        for X, y, shape in val_dataloader:
            shape = shape.numpy()
            y_pred = net(X)
            loss_value = loss(y_pred, y) 
            loss_history.append(loss_value.cpu().data.item())

        val_loss = np.mean(loss_history)
        print("val_loss:\t%.5f" % val_loss)
        print("val_quality:\t%.5f" % (val_loss * 100**2))
        
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
        print(f"iter={i}/{len(test_dataloader) - 1}", end='\r')
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

