import json
import pathlib
import pickle

import numpy as np
import sklearn.metrics
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.help_tools import make_logger, parse_args
from lib.metric_factory import MetricFactory
from lib.net_factory import NetFactory


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
    if logging_file := learning_process["logger"]["logging_file"]:
        logger.debug(
            f"stdout is duplicated to file '{logging_file}' as well."
        )

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
    logger.debug(f"{torch.device(device_identifier)=}")
    if device.type == "cuda":
        logger.debug(f"{torch.cuda.get_device_name(device_identifier)=}")
        logger.debug(f"{torch.cuda.get_device_properties(device)}")

    loss_params: dict = learning_process["hyper_params"]["loss"]
    optimizer_params: dict = learning_process["hyper_params"]["optimizer"]
    epoch_nums: int = learning_process["hyper_params"]["epoch_nums"]

    # init model from pretrained state / continue unfinished learning.
    net = NetFactory.create_network(net_arch)
    logger.debug("Model has been created successfully with no pretrain.")
    if learning_process["continue_from"]:
        checkpoint = torch.load(
            learning_process["continue_from"],
            map_location=device
        )
        net.load_state_dict(checkpoint['model_state_dict'])
        # TODO: add here other *_state_dict loads.
        epoch_start = checkpoint["epoch"] + 1
        logger.debug(
            "Checkpoint"
            f" `{learning_process['start_from']['checkpoint_path']}`"
            " have been successfully loaded!"
        )
    else:
        epoch_start = 0
    epoch_end = epoch_start + epoch_nums
    logger.debug(f"Last epoch of learning is {epoch_end} (not including).")

    net = net.to(device)
    logger.debug("Model has been transfered to device successfully.")
    loss = getattr(torch.nn, loss_params["loss_type"])(**loss_params["params"])
    logger.debug(f"Loss `{loss_params['loss_type']}` created successfully.")
    optimizer = getattr(torch.optim, optimizer_params["optimizer_type"])(
        net.parameters(),
        **optimizer_params["params"]
    )
    logger.debug(
        f"Optimizer `{optimizer_params['optimizer_type']}`"
        " has been created successfully."
    )

    writer = SummaryWriter(learning_process["tensorboard_logs"])
    logger.debug("SummaryWriter has been created successfully.")

    assert learning_process["main_metric_name"] in {
        metric["name"] for metric in learning_process["metrics"]
    }, f"Main metric {learning_process['main_metric_name']}"
    " wasn't finded among metrics!"

    metrics: tp.Dict[str, tp.Any] = {
        metric["name"]: MetricFactory.create_metric(
            metric["function"],
            metric["params"]
        )
        for metric in learning_process["metrics"]
    }
    logger.debug(
        "MetricFactory has created all metric handlers successfully!"
    )

    main_metric_name = learning_process["main_metric_name"]
    best_main_metric_value = metrics[main_metric_name].worst_value
    best_epoch = 0
    train_loss_in_the_best_epoch = float("inf")
    val_loss_in_the_best_epoch = float("inf")

    # training itself.
    for epoch in range(epoch_start, epoch_end, 1):
        logger.info(f"Epoch №{epoch} has started.")
        net.train()
        logger.debug("Model set to train mode.")
        loss_history = []
        for X, y, *_ in dataloaders["train"]:
            logger.debug("Data have been erased from dataloaders['train'].")
            X, y = X.to(device), y.to(device)
            logger.debug("X, y tensors have been transported to device.")
            optimizer.zero_grad()
            logger.debug("Gradient has been cleared by the optimizer.")
            y_pred = net(X)
            logger.debug("Network has infered and returned prediction.")
            loss_value = loss(y_pred, y)
            logger.debug("Loss value has been calculated.")
            loss_value.backward()
            logger.debug("Backward pass has been done.")
            optimizer.step()
            logger.debug("Optimizer has done a step.")
            cur_train_loss = loss_value.cpu().data.item()
            loss_history.append(cur_train_loss)
            logger.info("another train batch loss: %f" % cur_train_loss)

        train_loss = np.mean(loss_history)
        logger.info("train loss: %f" % train_loss)

        net.eval()
        logger.debug("Model has been set to eval mode.")
        loss_history: tp.List[float] = []
        y_true_history: tp.List[tp.List[float]] = []
        y_pred_history: tp.List[tp.List[float]] = []
        with torch.no_grad():
            for X, y, *_ in dataloaders["val"]:
                logger.debug("Data have been erased from dataloaders['val'].")
                X, y = X.to(device), y.to(device)
                logger.debug("X, y tensors have been transported to device.")
                y_pred = net(X)
                logger.debug("Network has infered and returned prediction.")
                loss_value = loss(y_pred, y)
                logger.debug("Loss value has been calculated.")
                loss_history.append(loss_value.cpu().data.item())
                y_true_history.extend(y.cpu().data.tolist())
                y_pred_history.extend(y_pred.cpu().data.tolist())
        val_loss = np.mean(loss_history)
        logger.info("val loss: %f" % val_loss)

        y_true_history: np.ndarray = np.array(y_true_history)
        y_pred_history: np.ndarray = np.array(y_pred_history)

        calculated_metrics: tp.Dict[str, float] = dict()
        for name, metric_handler in metrics.items():
            calculated_metrics[name] = metric_handler(
                y_true_history,
                y_pred_history
            )
            logger.info(
                f"`{name}` value at val:  %f" % calculated_metrics[name]
            )
        logger.debug("All metrics have been calculated!")

        writer.add_scalars(
            "Loss",
            {
                "train": train_loss,
                "val": val_loss
            },
            epoch
        )
        writer.add_scalars("Metrics", calculated_metrics, epoch)
        logger.debug("Writer have added new scalars.")

        if metrics[main_metric_name].is_it_better_than(best_main_metric_value):
            best_main_metric_value = calculated_metrics[main_metric_name]
            train_loss_in_the_best_epoch = train_loss
            val_loss_in_the_best_epoch = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric_value": best_main_metric_value,
                    "epoch": epoch
                },
                pathlib.Path(learning_process["best_model_dump"])
            )
            logger.info("New best dump has been achived and saved!")

        torch.save(
            {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_state_dict': loss.state_dict(),
                'epoch': epoch,
                'current_val_loss_value': val_loss,
                'best_epoch': best_epoch,
                'best_metric_value': best_main_metric_value
            },
            pathlib.Path(learning_process["last_model_dump"])
        )
        logger.debug("New last checkpoint has been saved!")

    logger.debug("Training completed.")
    logger.info(
        f"Best `{main_metric_name}` value ({best_main_metric_value})"
        f" was achived at the end of {best_epoch}'th epoch (by"
        " counting from zero) (the corresponding"
        f" train loss is {train_loss_in_the_best_epoch},"
        f" val loss is {val_loss_in_the_best_epoch})."
    )
    writer.close()
    logger.debug("Writer closed.")


if __name__ == "__main__":
    main()

