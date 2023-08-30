import argparse
import json
import pathlib
import typing as tp

import torch


def torch_module_to_json(
        name: str,
        module: torch.nn.Module
) -> tp.Dict[str, tp.Any]:
    if isinstance(module, torch.nn.AdaptiveAvgPool2d):
        return {
            "name": name,
            "layer_type": "AdaptiveAvgPool2d",
            "params": {
                "output_size": module.output_size
            }
        }
    elif isinstance(module, torch.nn.BatchNorm2d):
        return {
            "name": name,
            "layer_type": "BatchNorm2d",
            "params": {
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine,
                "track_running_stats": module.track_running_stats
            }
        }
    elif isinstance(module, torch.nn.Conv2d):
        return {
            "name": name,
            "layer_type": "Conv2d",
            "params": {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "groups": module.groups,
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "bias": module.bias is not None,
                "padding_mode": module.padding_mode
            }
        }
    elif isinstance(module, torch.nn.Linear):
        return {
            "name": name,
            "layer_type": "Linear",
            "params": {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None
            }
        }
    elif isinstance(module, torch.nn.MaxPool2d):
        return {
            "name": name,
            "layer_type": "MaxPool2d",
            "params": {
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "dilation": module.dilation,
                "return_indices": module.return_indices,
                "ceil_mode": module.ceil_mode
            }
        }
    elif isinstance(module, torch.nn.ReLU):
        return {
            "name": name,
            "layer_type": "ReLU",
            "params": {
                "inplace": module.inplace
            }
        }        
    else:
        assert False, name


def module_to_json(
        name: str,
        module: torch.nn.Module,
        freeze: bool
) -> tp.Dict[str, tp.Any]:
    if isinstance(
            module,
            tuple(
                getattr(torch.nn, module_name)
                for module_name in [
                    "AdaptiveAvgPool2d", 
                    "BatchNorm2d",
                    "Conv2d", 
                    "Linear", 
                    "MaxPool2d", 
                    "ReLU"
                ]
            )
    ):
        json_object = torch_module_to_json(name, module)
        if freeze:
            json_object.update(
                {
                    "post_process": {
                        "freeze_grad": True,
                        "pretrain": None,
                        "remove_layers": None
                    }
                }
            )
        return json_object
    elif isinstance(module, torch.nn.Sequential):
        return {
            "name": name,
            "layer_type": "Sequential",
            "params": [
                module_to_json(sub_name, sub_module, freeze)
                for sub_name, sub_module in module.named_children()
            ]
        }
    else:
        return {
            "name": name,
            "layer_type": module._get_name(),
            "params": {
                "architecture": [
                    module_to_json(sub_name, sub_module, freeze)
                    for sub_name, sub_module in module.named_children()
                ]
            }
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--freeze', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_path = pathlib.Path(args.model)
    save_to_path = pathlib.Path(args.save_to)
    model: torch.nn.Module = torch.load(model_path)
    architecture: tp.Dict[str, tp.Any] = module_to_json(
        name=model_path.name.rsplit(".", 1)[0],
        module=model,
        freeze=args.freeze
    )
    with open(args.save_to, "w") as f:
        json.dump(architecture, f, indent=4)

