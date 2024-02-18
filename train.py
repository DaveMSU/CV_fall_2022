import json
import pathlib
import pickle
from collections import defaultdict

import numpy as np
import sklearn.metrics
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.help_tools import make_logger, parse_args, RunningMeansHandler
from lib.metric_factory import MetricFactory
from lib.net_factory import NetFactory


def main():
    # config files loading.
    args = parse_args()
    net_arch_path = pathlib.Path(args.net)
    learning_process_path = pathlib.Path(args.learn)

    with open(net_arch_path, "r") as f:
        net_arch = json.load(f)

    with open(learning_process_path, "r") as f:
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
        with open(learning_process["data"][mode]["dump_path"], "rb") as f:
            dataloaders[mode] = DataLoader(
                dataset=pickle.load(f),
                batch_size=learning_process["data"][mode]["batch_size"],
                shuffle=learning_process["data"][mode]["shuffle"]
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
    lr_scheduler_params: dict = learning_process["hyper_params"]["lr_scheduler"]
    if "lr_lambda" in lr_scheduler_params["params"]:
        lr_scheduler_params["params"]["lr_lambda"] = eval(
            lr_scheduler_params["params"]["lr_lambda"]
        )
    assert lr_scheduler_params["use_after"] in ["step", "epoch"]
    epoch_nums: int = learning_process["hyper_params"]["epoch_nums"]
    visualize_outputs: tp.List[tp.Dict[str, tp.Union[str, int, tp.Callable[[int], bool]]]] = \
        learning_process["sub_net_outputs_to_visualize"]
    visualize_outputs: tp.List[tp.Dict[str, tp.Union[str, int, tp.Callable[[int], bool]]]] = [
        {
            key: eval(value) if key == "inclusion_condition" else value
            for key, value in output_config.items()
        } for output_config in learning_process["sub_net_outputs_to_visualize"]
    ]
    logger.debug(
        "Subnet outputs to visualize: " +
        ", ".join(
            map(lambda d: f'\"{d["sub_net_name"]}\"', visualize_outputs)
        ) + "."
    )

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
            f" `{learning_process['continue_from']}`"
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
    lr_scheduler = getattr(  # TODO: support SequentialLR here.
        torch.optim.lr_scheduler,
        lr_scheduler_params["lr_scheduler_type"]
    )(
        optimizer,
        **lr_scheduler_params["params"]
    )
    logger.debug(
        f"Scheduler of lr `{lr_scheduler_params['lr_scheduler_type']}`"
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
        "MetricFactory has created following metric handlers successfully: "
        + ", ".join([f"\"{metric_name}\"" for metric_name in metrics])
        + "."
    )

    main_metric_name = learning_process["main_metric_name"]
    best_main_metric_value: float = metrics[main_metric_name].worst_value
    logger.debug(
        f"Worst value ({best_main_metric_value}) as init has been "
        f"chosen for the main metric (\"{main_metric_name}\")."
    )

    best_epoch = 0
    train_loss_in_the_best_epoch = float("inf")
    val_loss_in_the_best_epoch = float("inf")

    # training itself.
    step: int = 0
    for epoch in range(epoch_start, epoch_end, 1):
        logger.info(f"Epoch №{epoch} has started.")
        net.train()
        logger.debug("Model has been set to `train` mode.")
        running_means: tp.Dict[str, tp.Union[dict, RunningMeansHandler]] = {
            key: RunningMeansHandler()
            for key in ["train_loss", "val_loss"]
        }
        running_means["metrics"] : tp.Dict[str, RunningMeansHandler] = {
            metric_name: RunningMeansHandler() for metric_name in metrics
        }
        running_means["grads"]: tp.Dict[str, RunningMeansHandler] = {
            name: RunningMeansHandler() for name, _ in net.named_children()
        }
        for X, y, *_ in dataloaders["train"]:
            logger.debug(
                "Data have been erased from dataloaders['train'] with shapes:"
                f" `X` ~ {tuple(X.shape)}, `y` ~ {tuple(y.shape)}."
            )
            X, y = X.to(device), y.to(device)
            logger.debug("X, y tensors have been transported to device.")
            optimizer.zero_grad()
            logger.debug("Gradient has been cleared by the optimizer.")
            y_pred = net(X)
            logger.debug(
                "Network has infered and returned prediction with shape:"
                f" `y_pred` ~ {tuple(y_pred.shape)}"
            )
            loss_value = loss(y_pred, y)
            logger.debug("Loss value has been calculated.")
            loss_value.backward()
            logger.debug("Backward pass has been done.")
            optimizer.step()
            logger.debug("Optimizer has done a step.")
            # TODO: try without .data
            cur_train_loss = loss_value.cpu().data.item()
            running_means["train_loss"].add(cur_train_loss, n=y.shape[0])
            logger.info("another train batch loss: %f" % cur_train_loss)
            for sub_net_name, sub_net in net.named_children():  # TODO: RAM memory bottle neck.
                gradient = np.array([])
                for param in sub_net.parameters():
                    grad_: tp.Optional[torch.Tensor] = param.grad
                    if grad_ is not None:
                        gradient = np.concatenate(
                            (gradient, grad_.view(-1).cpu().numpy())
                        )
                running_means["grads"][sub_net_name].add(
                    gradient, n=y.shape[0]
                )
            logger.debug("Averaged grad has been accumulated.")

            if lr_scheduler_params["use_after"] == "step":
                # TODO: incorrect when number of groups more than one.
                prev_learning_rate: float = optimizer.param_groups[-1]["lr"]
                if lr_scheduler_params["lr_scheduler_type"] == "ReduceLROnPlateau":
                    lr_scheduler.step(cur_train_loss)
                else:
                    lr_scheduler.step()
                cur_learning_rate: float = optimizer.param_groups[-1]["lr"]
                logger.debug("Step of lr_scheduler has been made.")
                logger.info(
                    "Learning rate has changed from value"
                    f" `{prev_learning_rate}` to `{cur_learning_rate}`."
                )
                writer.add_scalar("LearningRate", prev_learning_rate, step)
            step += 1

        grad_norms: tp.Dict[str, float] = {
            sub_net_name: np.power(gradient.get_value(), 2).sum() ** 0.5
            for sub_net_name, gradient in running_means["grads"].items()
        }
        logger.debug("All grad norms has been calculated.")
        logger.info(
            "train loss: %f" % running_means["train_loss"].get_value()
        )
        embeddings = {
            output_config["sub_net_name"]: dict(
                mat=None,
                metadata=[],
                label_img=torch.Tensor([]).to(device),
                global_step=-1,
                tag=output_config["sub_net_name"]
            ) for output_config in visualize_outputs
        }
        net.eval()
        logger.debug("Model has been set to `eval` mode.")
        with torch.no_grad():
            for X, y, *_ in dataloaders["val"]:
                logger.debug(
                    "Data have been erased from dataloaders['val'] with shapes:"
                    f" `X` ~ {tuple(X.shape)}, `y` ~ {tuple(y.shape)}."
                )
                X, y = X.to(device), y.to(device)
                logger.debug("X, y tensors have been transported to device.")
                all_stages = net.full_forward(X)  # OrderedDict[str, torch.Tensor]
                logger.debug(
                    "Network has infered and returned outputs of all stages."
                )
                for output_config in visualize_outputs:
                    sub_net: str = output_config["sub_net_name"]
                    max_vectors: int = output_config["number_of_vectors"]
                    if len(embeddings[sub_net]["metadata"]) == max_vectors:
                        continue
                    if not output_config["inclusion_condition"](epoch):
                        continue
                    old = embeddings[sub_net]["mat"]
                    new = all_stages[sub_net].cpu().numpy()
                    embeddings[sub_net]["mat"] = (
                        new if old is None else np.concatenate((old, new))
                    )[:max_vectors]  # TODO: try torch type, concat works better.
                    embeddings[sub_net]["metadata"] = (
                        embeddings[sub_net]["metadata"] + y.argmax(dim=1).cpu().tolist()
                    )[:max_vectors]
                    embeddings[sub_net]["label_img"] = torch.concat(
                        (embeddings[sub_net]["label_img"], X)
                    )[:max_vectors]
                    if embeddings[sub_net]["global_step"] != epoch:
                        embeddings[sub_net]["global_step"] = epoch
                    logger.debug(
                        f"It has been obtained embeddings from `{sub_net}`"
                        f" subnet with shape: {tuple(new.shape)}"
                    )
                logger.debug("Embeddings have been calculated without errors.")
                y_pred: torch.Tensor = all_stages.popitem(last=True)[1]
                logger.debug(
                    "Var `y_pred` has been taken from the last stage with"
                    f" shape: `y_pred` ~ {tuple(y_pred.shape)}"
                )
                loss_value = loss(y_pred, y)
                logger.debug("Loss value has been calculated.")
                running_means["val_loss"].add(
                    loss_value.cpu().data.item(), n=y.shape[-1]
                )
                for name, metric_handler in metrics.items():
                    # TODO: check if it works incorrectly for roc auc score.
                    running_means["metrics"][name].add(
                        metric_handler(y.cpu().numpy(), y_pred.cpu().numpy()),
                        n=y.shape[0]
                    )
                logger.debug("All metrics have been calculated!")
        logger.info("val loss: %f" % running_means["val_loss"].get_value())
        for metric_name in metrics:
            metric_value = running_means["metrics"][metric_name].get_value()
            logger.info(
                f"`{metric_name}` value at val:  %f" % metric_value
            )

        if lr_scheduler_params["use_after"] == "epoch":
            # TODO: incorrect when number of groups more than one.
            prev_learning_rate: float = optimizer.param_groups[-1]["lr"]
            if lr_scheduler_params["lr_scheduler_type"] == "ReduceLROnPlateau":
                lr_scheduler.step(running_means[main_metric_name].get_value())
            else:
                lr_scheduler.step()
            cur_learning_rate: float = optimizer.param_groups[-1]["lr"]
            logger.debug("Step of lr_scheduler has been made.")
            logger.info(
                "Learning rate has changed from value"
                f" `{prev_learning_rate}` to `{cur_learning_rate}`."
            )
            writer.add_scalar("LearningRate", prev_learning_rate, epoch)

        for embds in embeddings.values():
            if embds["global_step"] != -1:
                writer.add_embedding(**embds)
        logger.debug("Writer have added new embeddings.")
                
        writer.add_scalars(
            "Loss",
            {
                "train": running_means["train_loss"].get_value(),
                "val": running_means["val_loss"].get_value()
            },
            epoch
        )
        writer.add_scalars(
            "Metrics",
            {k: v.get_value() for k, v in running_means["metrics"].items()},
            epoch
        )
        writer.add_scalars("GradNorms", grad_norms, epoch)
        logger.debug("Writer have added new scalars.")

        if metrics[main_metric_name].is_first_better_than_second(
                running_means["metrics"][main_metric_name].get_value(),
                best_main_metric_value
        ):
            best_main_metric_value: float = \
                running_means["metrics"][main_metric_name].get_value()
            train_loss_in_the_best_epoch: float = running_means["train_loss"].get_value()
            val_loss_in_the_best_epoch: float = running_means["val_loss"].get_value()
            best_epoch: int = epoch
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
                'current_val_loss_value': running_means["val_loss"],
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
