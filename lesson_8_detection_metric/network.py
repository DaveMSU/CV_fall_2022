import hashlib
import typing as tp

import numpy as np
import torch


class ClassifyNetwork(torch.nn.Module):
    def __init__(
            self,
            base_net: torch.nn.Module,
            internal_features: int = 1024,
            output_classes_num: int = 205,
            correct_priors: tp.Optional[tp.Dict[str, float]] = None
    ):
        super(ClassifyNetwork, self).__init__()
        if correct_priors is None:
            correct_priors = {
                str(i): 1 / output_classes_num
                    for i in range(output_classes_num)
            }
        assert output_classes_num == len(correct_priors)
        assert len(set(correct_priors)) == output_classes_num
        assert list(correct_priors)[0] == "0"
        assert list(correct_priors)[-1] == str(output_classes_num - 1)
        assert np.isclose(sum(correct_priors.values()), 1.0)
        self._p_real_priori = np.array(list(correct_priors.values()))

        in_features_at_last_fc = list(base_net.children())[-1].in_features
        base_net.fc = torch.nn.Linear(
            in_features = in_features_at_last_fc,
            out_features = internal_features
        )
        self._backbone = torch.nn.Sequential(
            base_net,
            torch.nn.ReLU()
        )
        
        first_layers_number_to_be_frozen = len(list(self._backbone[0].children())) - 1
        for layer_id, layer in enumerate(self._backbone[0].children()):
            if layer_id < first_layers_number_to_be_frozen:
                for param in layer.parameters():
                    param.requires_grad = False

        self._head = torch.nn.Linear(
            in_features=internal_features,
            out_features=output_classes_num
        )

        self._hash_map = {
            "x_hash": "",  # x to embedding
            "embs_hash": "",  # x to embedding
            "embs": torch.Tensor(),  # embedding to y_logits
            "y_logits": torch.Tensor()  # embedding to y_logits
        }

        self._output_classes_num = output_classes_num

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        x_hash: str = hashlib.sha256(str(x.cpu().data).encode()).hexdigest()
        if x_hash == self._hash_map["x_hash"]:
            embeddings = self._hash_map["embs"]
        else:
            embeddings = self._backbone(x)
            self._hash_map["x_hash"] = x_hash
            self._hash_map["embs"] = embeddings
        return embeddings

    def head(self, embs: torch.Tensor) -> torch.Tensor:
        embs_hash: str = hashlib.sha256(str(embs.cpu().data).encode()).hexdigest()
        if embs_hash == self._hash_map["embs_hash"]:
            y_logits = self._hash_map["y_logits"]
        else:
            y_logits = self._head(embs)
            self._hash_map["embs_hash"] = embs_hash
            self._hash_map["y_logits"] = y_logits
        return y_logits
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone(x)
        y_logits = self.head(embeddings)
        return y_logits
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        y_logits = self(x)
        y_probs_torch: torch.Tensor = torch.nn.functional.softmax(y_logits, dim=1)
        
        p_learned_priori: float = 1 / self._output_classes_num
        p_learned_posteriori: np.ndarray = y_probs_torch.cpu().data.numpy()
        
        p_real_posteriori_unnormed = p_learned_posteriori * self._p_real_priori / p_learned_priori
        p_real_posteriori = p_real_posteriori_unnormed / p_real_posteriori_unnormed.sum(axis=1).reshape(-1, 1)
        
        assert p_real_posteriori.shape == p_learned_posteriori.shape
        assert np.isclose(p_real_posteriori.sum(axis=1).max(), 1.0)
        assert np.isclose(p_real_posteriori.sum(axis=1).min(), 1.0)
        return p_real_posteriori
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        y_probs = self.predict_proba(x)
        y_pred = y_probs.argmax(axis=1)
        return y_pred

