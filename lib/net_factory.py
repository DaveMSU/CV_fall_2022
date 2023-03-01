import typing as tp

import torch


# def recursion_getattr(item: tp.Any, full_name: str) -> tp.Callable:
#     for name_part in full_name.split("."):
#         item = getattr(item, name_part)
#     return item
# 
# 
# def recursion_hasattr(item: tp.Any, full_name: str) -> bool:
#     for name_part in full_name.split("."):
#         if not hasattr(item, name_part):
#             return False
#         item = getattr(item, name_part)
#     return True


class Flatten(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Pad(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._params = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, **self._params)
    

class NeuralNetwork(torch.nn.Module):  
    def __init__(self, architecture: tp.List[tp.Dict[str, tp.Any]]):
        super().__init__()
        self._layers_seq = torch.nn.Sequential(
            *[
                getattr(torch.nn, layer["layer_type"])(**layer["params"]) 
                    if hasattr(torch.nn, layer["layer_type"]) 
                    else globals()[layer["layer_type"]](**layer["params"])
                for layer in architecture
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._layers_seq(x)

