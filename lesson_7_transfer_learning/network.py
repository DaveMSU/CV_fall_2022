import torch


class BirdNet(torch.nn.Module):
    def __init__(
            self,
            base_net: torch.nn.Module,
            output_classes_num: int = 50,
            first_layers_number_to_be_frozen: int = 9
    ):
        super().__init__()
        assert first_layers_number_to_be_frozen >= 0,\
            "Variable freeze_first do not supports of negative numbers."
        
        self._base_net = base_net
        in_features_at_last_fc = list(self._base_net.children())[-1].in_features
        self._base_net.fc = torch.nn.Linear(
            in_features = in_features_at_last_fc,
            out_features = output_classes_num
        )
        
        for layer_id, layer in enumerate(self._base_net.children()):
            if layer_id < first_layers_number_to_be_frozen:
                for param in layer.parameters():
                    param.requires_grad = False
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._base_net.forward(x)
    
