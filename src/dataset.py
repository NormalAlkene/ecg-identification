"""Dataset class for loading data from the dataset.
"""

from typing import override
from multiprocessing import Process, SimpleQueue

import numpy as np
from numpy import ndarray
import torch
from torch import Tensor
from torch.utils.data import Dataset
import lightning as pl

class EcgIdDataset(Dataset):
    """ECG identification dataset
    """
    _device: str
    _dtype: torch.dtype
    _sample_len: int

    _data_raw:  list[tuple[ndarray, str]]
    """data
    [(ndarray[sample_len], person_id), ...]
    """
    _sample_list: list[tuple[int, int]]
    """_sample_list
    [(signal_idx, start_pos), ...]
    """

    def __init__(self, path: str, sample_len: int, token_len: int, device: str = 'cpu', dtype: torch.dtype = torch.float32):
        super().__init__()
        self._device = device
        self._dtype = dtype
        self._sample_len = sample_len
        self._token_len = token_len
        self._data_raw = []
        with np.load(path) as data:
            self._data_raw = [ (signal, k) for k, v in data.items() for signal in v ]

        self.resample()

    def __len__(self) -> int:
        return len(self._sample_list) * (len(self._sample_list) - 1)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        idx_first = idx // (len(self._sample_list) - 1)
        idx_second = idx % (len(self._sample_list) - 1)
        # avoid same index
        if idx_second >= idx_first:
            idx_second += 1

        cur_data = (self._get_sample(idx_first), self._get_sample(idx_second))
        label = torch.tensor(
            [cur_data[0][1] == cur_data[1][1], cur_data[0][2] == cur_data[1][2]],
            device = self._device,
            dtype = self._dtype
            )
        return cur_data[0][0], cur_data[1][0], label

    def _get_sample(self, idx: int) -> tuple[Tensor, str, int]:
        """Get a sample of signal

        Args:
            idx (int): sample index

        Raises:
            RuntimeError: invalid token length

        Returns:
            tuple[Tensor, str, int]: signal, person index, signal index
        """
        signal_idx, start_pos = self._sample_list[idx]
        data, person_idx = self._data_raw[signal_idx]
        data = torch.tensor(data[start_pos : start_pos + self._sample_len],
                            device = self._device, dtype = self._dtype)
        data = data.view(-1, self._token_len)
        return data, person_idx, signal_idx

    def gen_resample_list(self) -> list[tuple[int, int]]:
        """Generate resample list

        Args:
            num_resample (int): number of resample

        Returns:
            list[tuple[int, int]]: signal index, start position
        """
        ret: list[tuple[int, int]] = []
        for idx, (signal, _) in enumerate(self._data_raw):
            sample_num = signal.size // self._sample_len
            # total minimum space
            if signal.size % self._sample_len <= 0.5 * self._sample_len:
                sample_num -= 1

            total_space = signal.size - self._sample_len * sample_num

            spaces = np.random.uniform(0, 1, size = sample_num + 1)
            spaces = np.round(spaces / spaces.sum() * total_space).astype(int)
            start_poses = np.arange(sample_num) * self._sample_len + spaces.cumsum()[:-1]
            ret.extend(zip([idx] * sample_num, start_poses.tolist()))

        return ret

    def resample(self, resample_list: list[tuple[int, int]] | None = None) -> None:
        """Resample the dataset
        """
        if resample_list is None:
            resample_list = self.gen_resample_list()
        self._sample_list = resample_list

class ResampleCallback(pl.Callback):
    """Resample callback
    """
    _q_training: SimpleQueue
    _q_validation: SimpleQueue

    def __init__(self):
        super().__init__()
        self._q_training = SimpleQueue()
        self._q_validation = SimpleQueue()

    def _worker(self, dataset: EcgIdDataset, queue: SimpleQueue) -> None:
        queue.put(dataset.gen_resample_list())

    @override
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # TODO: check
        Process(
            target = self._worker,
            args = (trainer.train_dataloader.dataset, self._q_training)
        ).start()
        Process(
            target = self._worker,
            args = (trainer.val_dataloaders.dataset, self._q_validation)
        )

    @override
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        pl_module.train_dataset.resample(self._q_training.get())
        pl_module.val_dataset.resample(self._q_validation.get())
