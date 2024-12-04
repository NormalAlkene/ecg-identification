#!/usr/bin/env python3
"""
The config file of ECG identification
"""

from dataclasses import dataclass

@dataclass
class _paths:
    path_ds_raw: str = 'data/0-raw'
    path_ds_preprocessed: str = 'data/1-preprocessed'
    path_ds_training: str = 'data/2-training'
    path_ds_validation: str = 'data/2-validation'
    path_ds_testing: str = 'data/2-testing'
    path_results: str = 'data/3-results'

PATHS = _paths()

@dataclass
class _preprocessing:
    fs: float = 512.0
    detrend_order: int = 9
    lowcut: float = 0.5
    highcut: float = 256 # of no use
    ratio_validation: float = 0.15
    ratio_testing: float = 0.15

PREPROCESSING = _preprocessing()
