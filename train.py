import json
import pathlib
import pickle
import typing as tp

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.datasets import *
from lib.help_tools import make_logger, parse_args
from lib.net_factory import NeuralNetwork


def main():
    # config files loading.
    args = parse_args()
    net_arch_path = pathlib.Path(args.net)
    learning_process_path = pathlib.Path(args.learn)
 
    with open(net_arch_path) as f:
        net_arch = json.load(f)
    
    with open(learning_process_path) as f:
        learning_process = json.load(f)
    
    logger = make_logger(**learning_process["logger"])
    logger.debug("Logger created.")

    # train/val dataloaders loading.
    dataloaders = dict()
    for mode in ["train", "val"]:
        with open(learning_process["data"][f"{mode}"]["dump_path"], "rb") as f:
            dataloaders[mode] = DataLoader(
                dataset=pickle.load(f),
                batch_size=learning_process["data"][f"{mode}"]["batch_size"],
                shuffle=learning_process["data"][f"{mode}"]["shuffle"]
            )
        logger.debug(
            f"{mode.capitalize()} dataloader uploaded successfully."
        )

    # configation of learning process.
    device_identifier: str = learning_process["device"]
    device = torch.device(device_identifier)
    logger.debug(
        f'"{device_identifier}" device was selected successfully!'
    )
    logger.debug(f"{torch.__version__=}")
    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.cuda.is_available()=}")
    logger.debug(f"{torch.cuda.device_count()=}")
    logger.debug(f"{torch.cuda.current_device()=}")
    logger.debug(f"{device_identifier=}")
    logger.debug(f"{torch.cuda.get_device_name(device_identifier)=}")
    logger.debug(f"{torch.device(device_identifier)=}")
    logger.debug(f"{torch.cuda.get_device_properties(device)}")

    loss_params = learning_process["hyper_params"]["loss"]
    optimizer_params = learning_process["hyper_params"]["optimizer"]
    epoch_nums = learning_process["hyper_params"]["epoch_nums"]

    net = NeuralNetwork(net_arch)
    logger.debug("Model created successfully with no pretrain.")
    if learning_process["start_model"]:
        checkpoint = torch.load(learning_process["start_model"])
        net.load_state_dict(checkpoint['model_state_dict'])
        logger.debug("Pretrain init state loaded.")
    net = net.to(device)
    logger.debug("Model transfered to device successfully.")
    loss = getattr(torch.nn, loss_params["loss_type"])(**loss_params["params"])
    logger.debug(f"Loss `{loss_params['loss_type']}` created successfully.")
    optimizer = getattr(torch.optim, optimizer_params["optimizer_type"])(
        net.parameters(),
        **optimizer_params["params"]
    )
    logger.debug(
        f"Optimizer `{optimizer_params['optimizer_type']}` created successfully."
    )
    writer = SummaryWriter(learning_process["tensorboard_logs"])
    logger.debug("SummaryWriter created successfully.")
    best_val_loss, best_epoch = float("inf"), 0

    # training itself.
    for epoch in range(epoch_nums):
        logger.info(f"Epoch №{epoch} has started.")
        net.train()
        logger.debug("Model set to train mode.")
        loss_history = []
        for X, y, *_ in dataloaders["train"]:
            logger.debug("All data erased from dataloaders['train'].")
            X, y = X.to(device), y.to(device)
            logger.debug("X, y tensors transported to device.")
            optimizer.zero_grad()
            logger.debug("Gradient cleared by optimizer.")
            y_pred = net(X)
            logger.debug("Network infered and returned prediction.")
            loss_value = loss(y_pred, y)
            logger.debug("Loss value counted.")
            loss_value.backward()
            logger.debug("Backward pass done.")
            optimizer.step()
            logger.debug("Optimizer has done a step.")
            cur_train_loss = loss_value.cpu().data.item()
            loss_history.append(cur_train_loss)
            logger.info(f"another train batch loss: %f" % cur_train_loss)

        train_loss = np.mean(loss_history)
        logger.info(f"train loss: %f" % train_loss)

        net.eval()
        logger.debug("Model set to eval mode.")
        loss_history = []
        with torch.no_grad():
            for X, y, *_ in dataloaders["val"]:
                logger.debug("All data erased from dataloaders['val'].")
                X, y = X.to(device), y.to(device)
                logger.debug("X, y tensors transported to device.")
                y_pred = net(X)
                logger.debug("Network infered and returned prediction.")
                loss_value = loss(y_pred, y) 
                logger.debug("Loss value counted.")
                loss_history.append(loss_value.cpu().data.item())

        val_loss = np.mean(loss_history)
        logger.info("val loss: %f" % val_loss)
        logger.info("val quality: %f" % (val_loss * 100**2))
        
        writer.add_scalars(
            'Loss', 
            {
                'train': train_loss,
                'val': val_loss
            },
            epoch
        )
        logger.debug("Writer have added new scalars.")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                pathlib.Path(learning_process["model_dump"])
            )
            logger.info("New best dump has been achived!")

    logger.debug("Training completed.")
    logger.info(
        f"Best val loss ({best_val_loss}) was achived at the end of"\
        f" {best_epoch}'th epoch (by counting from zero)."
    )
    writer.close()
    logger.debug("Writer closed.")
    


if __name__ == "__main__":
    main()

