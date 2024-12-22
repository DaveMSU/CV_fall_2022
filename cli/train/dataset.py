import h5py
import pathlib

import torch

from lib import ModelInputOutputPairSample


class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, hdf5_file_path: pathlib.Path):
        self._hdf5_file = h5py.File(hdf5_file_path, "r")
        self._hdf5_ds_in = self._hdf5_file["input"]
        self._hdf5_ds_out = self._hdf5_file["output"]

    def __len__(self):
        assert len(self._hdf5_ds_in) == len(self._hdf5_ds_out)
        return len(self._hdf5_ds_in)

    def __getitem__(self, index: int) -> ModelInputOutputPairSample:
        sample = ModelInputOutputPairSample(
            torch.as_tensor(self._hdf5_ds_in[index]).cpu(),
            torch.as_tensor(self._hdf5_ds_out[index]).cpu()
        )
        # TODO: think about using custom Dataloader as well
        return sample.input, sample.output

    def __del__(self):
        self._hdf5_file.close()
