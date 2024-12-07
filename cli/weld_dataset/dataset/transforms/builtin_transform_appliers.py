import typing as tp

import torchvision

from .. import raw_sample_pair_handlers


class ModelInputBuiltInTransformApplier:  # TODO: inherit from an interface
    def __init__(self, transform_type: str, params: tp.Dict[str, tp.Any]):
        self._transform = getattr(
            torchvision.transforms,
            transform_type
        )(
            **params
        )

    def __call__(
            self,
            sample: raw_sample_pair_handlers.BaseRawModelInputOutputPairSample
    ) -> None:
        sample.input = self._transform(sample.input)
