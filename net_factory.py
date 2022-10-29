import typing as tp

import torch


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

