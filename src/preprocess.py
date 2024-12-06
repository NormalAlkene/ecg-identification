#!/usr/bin/env python3
"""
Preprocess
"""

import os
import shutil
import random
import multiprocessing
import argparse
from typing import Callable

import numpy as np
import pandas as pd
import scipy

from utils.fileUtils import traverse_file
from configs import PATHS, PREPROCESSING

def main(path_src: str = PATHS.path_ds_raw,
         path_dst: str = PATHS.path_ds_preprocessed,
         do_distribute: bool = True) -> None:
    """
    1. Load: load the .mat file;
    2. Preprocess: detrend and denoise (cut off in the frequency domain);
    3. Preprocess: normalize;
    4. Save: save the preprocessed data as .csv;
    5. Distribute: distribute the preprocessed data into train, validation and test sets;
    6. Save: save the distributed data as .npz.
    """
    # Preprocess
    file_paths = list(traverse_file(path_src, ".mat"))
    signals = [ load_mat(os.path.join(d, f)) for d, f in file_paths ]

    with multiprocessing.Pool() as pool:
        data = pool.map(Preprocessor(), signals)
    del signals

    # Group data by dir
    grouped: dict[str, list[np.ndarray]] = {}
    for idx, (directory, _) in enumerate(file_paths):
        cur_file = os.path.relpath(directory, path_src)
        if cur_file not in grouped:
            grouped[cur_file] = []
        grouped[os.path.relpath(directory, path_src)].append(data[idx])

    # Save data
    shutil.rmtree(path_dst, ignore_errors=True)
    os.makedirs(path_dst)
    for key, value in grouped.items():
        num_col = len(value)
        columns = ['t'] + [ str(i) for i in range(num_col)]

        t = np.arange(value[0].size) / PREPROCESSING.fs
        value.insert(0, t)

        df = pd.DataFrame(np.column_stack(value), columns = columns) # stacked by column
        df.to_csv(os.path.join(path_dst, f"{key}.csv"), index = False)

    # Distribute
    def distribute(total: list[str], path: str, num: int) -> None:
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        cur_dataset = random.sample(total, num)
        for file in cur_dataset:
            shutil.copy(os.path.join(path_dst, f"{file}.csv"),
                        os.path.join(path, f"{file}.csv"))
        total = list(set(total) - set(cur_dataset))

    USE_COLS = lambda x: x not in ["t", "f", "fft", "freq"]
    if do_distribute:
        datasets = list(grouped.keys()) # already relpath
        len_total = len(datasets)
        distribute(datasets, PATHS.path_ds_validation, int(len_total * PREPROCESSING.ratio_validation))
        csvdir_to_npz(PATHS.path_ds_validation,
                      os.path.join(PATHS.path_ds_validation, PATHS.name_ds_validation),
                      USE_COLS)
        distribute(datasets, PATHS.path_ds_testing, int(len_total * PREPROCESSING.ratio_testing))
        csvdir_to_npz(PATHS.path_ds_testing,
                      os.path.join(PATHS.path_ds_testing, PATHS.name_ds_testing),
                      USE_COLS)
        distribute(datasets, PATHS.path_ds_training, len(datasets))
        csvdir_to_npz(PATHS.path_ds_training,
                      os.path.join(PATHS.path_ds_training, PATHS.name_ds_training),
                      USE_COLS)
    else:
        csvdir_to_npz(path_dst, os.path.join(path_dst, "data.npz"), USE_COLS)

def load_mat(file_path: str) -> np.ndarray:
    """
    Load .mat file and return the data as numpy array.
    """
    return scipy.io.loadmat(file_path)["segmentData"].flatten()

class Detrender:
    """Detrender
    Detrend the data with np.polyfit.
    """
    def __init__(self, order: int = 1):
        self.order = order

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Detrend the data with np.polyfit.
        """
        coeffs = np.polyfit(np.arange(data.shape[0]), data, self.order)
        trend = np.polyval(coeffs, np.arange(data.shape[0]))
        return data - trend

class Denoiser:
    """Denoiser
    Denoise the data with FFT and IFFT
    """
    def __init__(self, fs, lowcut, highcut):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Cut off the data in the frequency domain.
        """
        freq = np.fft.fft(data)
        f = np.fft.fftfreq(data.shape[0], 1/self.fs)
        mask = (np.abs(f) < self.lowcut) | (np.abs(f) > self.highcut)
        freq[mask] = 0
        return np.fft.ifft(freq).real

class Normalizer:
    """Normalizer
    """
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize the data.
        """
        return (data - np.mean(data)) / np.std(data)

class Preprocessor:
    """Preprocessor
    """
    def __init__(self):
        self.preprocessors = [
            Detrender(order = PREPROCESSING.detrend_order),
            Denoiser(fs = PREPROCESSING.fs,
                     lowcut = PREPROCESSING.lowcut,
                     highcut = PREPROCESSING.highcut),
            Normalizer(),
        ]
    def __call__(self, data: np.ndarray) -> np.ndarray:
        for p in self.preprocessors:
            data = p(data)
        return data

def csvdir_to_npz(path_src: str, file_dst: str, use_cols: list[str] | Callable[[str], bool]) -> None:
    """
    Convert csv files in a directory to a npz file.
    """
    datasets: dict[str, np.ndarray] = {}
    for d, f in traverse_file(path_src, ".csv"):
        cur_data: pd.DataFrame = pd.read_csv(os.path.join(d, f), usecols = use_cols, index_col = None)
        datasets[f] = cur_data.to_numpy().transpose()
    np.savez_compressed(file_dst, **datasets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Preprocess")

    parser.add_argument("-i", "--input", type = str)
    parser.add_argument("-o", "--output", type = str)
    parser.add_argument("--no-distribute", action = "store_true")

    args = parser.parse_args()
    main(
        args.input if args.input else PATHS.path_ds_raw,
        args.output if args.output else PATHS.path_ds_preprocessed,
        not args.no_distribute
    )
