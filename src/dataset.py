"""Dataset class for loading data from the dataset.
"""

from typing import override
from itertools import product
import random

import numpy as np
from numpy.lib.npyio import NpzFile
import torch
from torch import Tensor
from torch.utils.data import Dataset
import lightning as pl

class EcgIdDataset(Dataset):
    """ECG identification dataset
    """
    _dtype: torch.dtype
    _sample_len: int
    _force_balance: bool

    _data_raw:  list[tuple[Tensor, str]]
    """data
    [(Tensor[n, sample_len], person_id), ...]
    """

    _sample_list: list[tuple[int, int, int]]
    """_sample_list
    [(person_idx, signal_idx, start_pos), ...]
    """

    _sample_block_list: list[int]
    """sample block list: the start of each sample block

    Not containing 0, but containing the end.

    [samp_idx, ...]
    """

    _combination_list: list[tuple[int, int]]
    """_combination_list
    [(samp_idx1, samp_idx2), ...]
    """

    def __init__(self,
                 data: str | tuple[str, str] | NpzFile | tuple[NpzFile, NpzFile],
                 sample_len: int,
                 token_len: int,
                 dtype: torch.dtype = torch.float32,
                 force_balance: bool = False):
        super().__init__()
        self._dtype = dtype
        self._sample_len = sample_len
        self._token_len = token_len
        self._force_balance = force_balance
        self._data_raw = []
        with np.load(data) as d:
            self._data_raw = [ (torch.tensor(v, dtype = dtype), k) for k, v in d.items() ]

        self.resample()
        self.recombine()

    def __len__(self) -> int:
        return len(self._combination_list)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        sample_idx = self._combination_list[idx]
        cur_data = [self.get_sample(i) for i in sample_idx]
        label = torch.tensor([
            cur_data[0][1] == cur_data[1][1], # person
            cur_data[0][1] == cur_data[1][1] and cur_data[0][2] == cur_data[1][2] # signal
        ], dtype = self._dtype)
        return cur_data[0][0], cur_data[1][0], label

    def get_sample(self, idx: int) -> tuple[Tensor, int, int]:
        """Get a sample of signal

        Args:
            idx (int): sample index

        Returns:
            tuple[Tensor, int, int]: signal, person index, signal index
        """
        person_idx, signal_idx, start_pos = self._sample_list[idx]
        data = self._data_raw[person_idx][0][signal_idx]
        data = data[start_pos : start_pos + self._sample_len].view(-1, self._token_len)
        return data, person_idx, signal_idx

    def combination_to_idx(self, idx: int) -> tuple[int, int]:
        """Convert combination index to sample index

        Args:
            idx (int): combination index

        Returns:
            tuple[int, int]: sample index pair
        """
        return self._combination_list[idx]

    def idx_to_rawname(self, idx: int) -> tuple[str, int]:
        """Get raw name by index

        Args:
            idx (int): sample index

        Returns:
            tuple[str, int]: person id, signal index
        """
        person_idx, signal_idx, _ = self._sample_list[idx]
        return self._data_raw[person_idx][1], signal_idx

    def resample(self) -> None:
        """Generate resample list

        Set variables:
            _sample_list: list[tuple[int, int, int]]: person index, signal index, start position
            _len_list:    list[int]: number of samples for each person
        """
        sample_list: list[tuple[int, int, int]] = []
        sample_block_list: list[int] = []
        for person_idx, (person, _) in enumerate(self._data_raw):
            for signal_idx, signal in enumerate(person):
                sample_num = len(signal) // self._sample_len
                # total minimum space
                if len(signal) % self._sample_len <= 0.2 * self._sample_len:
                    sample_num -= 1

                total_space = len(signal) - self._sample_len * sample_num
                spaces = np.random.uniform(0, 1, size = sample_num + 1)
                spaces = np.floor(spaces / spaces.sum() * total_space).astype(int)
                start_poses = np.arange(sample_num) * self._sample_len + spaces.cumsum()[:-1]
                sample_list.extend(zip(
                    [person_idx] * sample_num,
                    [signal_idx] * sample_num,
                    start_poses.tolist()
                ))
            sample_block_list.append(len(sample_list))

        self._sample_list = sample_list
        self._sample_block_list = sample_block_list

    def recombine(self) -> None:
        """Generate combination list

        Set variables:
            _combination_list: list[tuple[int, int]]: sample index pair
        """

        sample_set = set(range(len(self._sample_list)))
        comb_list: list[int] = []
        if self._force_balance:
            idx_start = 0
            for idx_end in self._sample_block_list:
                # the same block
                comb_list.extend(product(range(idx_start, idx_end), range(idx_start, idx_end)))
                # different blocks
                selected_samples = random.sample(
                    sorted(sample_set - set(range(idx_start, idx_end))),
                    idx_end - idx_start
                )
                comb_list.extend(product(range(idx_start, idx_end), selected_samples))

                idx_start = idx_end
        else:
            comb_list = product(range(len(self._sample_list)), range(len(self._sample_list)))

        self._combination_list = comb_list

class ResampleCallback(pl.Callback):
    """Resample callback
    """
    @override
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.train_dataloader.dataset.resample()
        trainer.train_dataloader.dataset.recombine()
        #trainer.val_dataloaders.dataset.resample()
