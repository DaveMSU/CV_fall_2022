import typing as tp
from collections import OrderedDict
from functools import reduce

import torch


def _dict_to_module(architecture: tp.Dict[str, tp.Any]) -> torch.nn.Module:
    params: tp.Union[list, dict] = architecture["params"]
    layer_type: str = architecture["layer_type"]

    if hasattr(torch.nn, layer_type):
        if layer_type == "Sequential" and isinstance(params, list):
            params = OrderedDict(
                { 
                    param["name"]: _dict_to_module(param)
                    for param in params
                }
            )
            module = getattr(torch.nn, layer_type)(params)
        else:
            assert isinstance(params, dict), architecture["name"]
            module =  getattr(torch.nn, layer_type)(**params)
    elif layer_type in globals():
        assert isinstance(params, dict), architecture["name"]
        module = globals()[layer_type](**params)
    else:
        assert False, architecture["name"]

    if "post_process" in architecture:
        if architecture["post_process"]["freeze_grad"]:
            for param in module.parameters():
                param.requires_grad = False
        if architecture["post_process"]["pretrain"] is not None:
            pretrain = torch.load(architecture["post_process"]["pretrain"])
            module.load_state_dict(pretrain["model_state_dict"])
        if architecture["post_process"]["remove_layers"] is not None:
            assert hasattr(module, "remove_layer")
            for layer_name in architecture["post_process"]["remove_layers"]:
                module.remove_layer(layer_name)
    return module


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Pad(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, **self._params)


class _BaseNetModule(torch.nn.Module):
    def __init__(self, architecture: tp.List[tp.Dict[str, tp.Any]]):
        super().__init__()
        self._modules_order: tp.List[str] = []
        for sub_architecture in architecture:
            self._modules_order.append(sub_architecture["name"])
            module = _dict_to_module(sub_architecture)
            setattr(self, sub_architecture["name"], module)
 
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def remove_layer(self, layer_name: str) -> None:
        self._modules_order.remove(layer_name)
        del self._modules[layer_name]


class BasicBlock(_BaseNetModule):
    def __init__(self, *args, **kwargs):
        self.downsample: tp.Optional[torch.nn.Module] = None
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(_BaseNetModule):
    def __init__(self, *args, **kwargs):
        self.downsample: tp.Optional[torch.nn.Module] = None
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class NeuralNetwork(_BaseNetModule):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return reduce(
            lambda x, f: f(x),
            [
                input_tensor,
                *[getattr(self, name) for name in self._modules_order]
            ]
        )


class NetFactory:
    @staticmethod
    def create_network(
            config: tp.Dict[str, tp.Any]
    ) -> torch.nn.Module:
        return _dict_to_module(config)

