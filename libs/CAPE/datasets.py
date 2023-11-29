""" Children of torch.utils.data.Dataset for the CAPE project. """

import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from kit.log import log_info, log_caller
from kit.path import join
from kit.bioinf import seq_to_idx, AA1_TO_IDX
from kit.bioinf.fasta import read_fasta, write_fasta
from kit.data import Split, DD


class FastaDS(Dataset):
    """Dataset that serves sequences from fasta files

    The dataset is loaded from a fasta file. The path to this fasta file
    is the concatenation of the input_directory_path and the path
    specified in the dhparams. The dhparams must contain the following
    attributes:
    - FOLDER: str - Name of the folder containing the fasta files
    - TRAIN: str - Name of the fasta file containing the training sequences
    - VAL: str - Name of the fasta file containing the validation sequences
    - TEST: str - Name of the fasta file containing the test sequences
    - BLOCK_SIZE: int - Size of the blocks that are returned by the dataset
    - WEIGHTED: boolean - Whether the dataset should be weighted
        by how often a sequence occurs in the dataset

    :param split: Split - Split of the dataset that should be loaded
    :param input_directory_path: str - Path to the input directory
    :param dhparams: DD - Data handling parameters
    :param mask: boolean - Whether to return a mask for the sequence
        1... if the token has a value
        0... if the token is a padding token
    """

    def __init__(self, split, input_directory_path, dhparams, mask=False):
        super().__init__()
        log_caller(self)

        if str(split) in dhparams:
            input_file = os.path.join(input_directory_path, dhparams[str(split)])
            if os.path.exists(input_file):
                self.df = read_fasta(input_file, return_df=True, evaluate=True)
                self.df["weight"] = self.df.shape[0] * self.df.cnt / self.df.cnt.sum()
                log_info(f"Loaded {self.df.shape[0]} samples for the {split} set")
            else:
                log_info(f"File {input_file} does not exist")
                self.df = None
        else:
            self.df = None

        self.block_size = dhparams.BLOCK_SIZE if "BLOCK_SIZE" in dhparams else None
        self.weighted = dhparams.WEIGHTED
        self.mask = mask

        log_info(f"block_size: {self.block_size} ")
        log_info(f"weighted: {self.weighted} ")

    def __len__(self):
        """returns the number of samples in the dataset"""

        return len(self.df) if self.df is not None else 0

    def __getitem__(self, j):
        """returns the j-th sample of the dataset"""

        row = self.df.iloc[j]
        seq = "*" + str(row.name)
        weight = row.weight if self.weighted else 1.0

        if self.block_size is not None and self.block_size < len(seq):
            start = random.randint(0, len(seq) - self.block_size - 1)
            seq = seq[
                start : (start + self.block_size + 1)
            ]  # a returned sequence must have block_size + 1 elements (the last is only a target)

        result = {
            "x": {"seq": torch.tensor(seq_to_idx(seq)), "weight": torch.tensor(weight)}
        }
        return result

    def collate(self, samples):
        """collates a list of samples into a batch"""

        max_length = np.max([sample["x"]["seq"].shape[-1] for sample in samples])

        batch = {
            "x": {
                "seq": torch.stack(
                    [
                        F.pad(
                            sample["x"]["seq"],
                            (0, max_length - sample["x"]["seq"].shape[-1]),
                            mode="constant",
                            value=AA1_TO_IDX["-"],
                        )
                        for sample in samples
                    ],
                    dim=0,
                ),
                "weight": torch.stack([sample["x"]["weight"] for sample in samples]),
            }
        }
        if self.mask:
            batch["x"]["mask"] = torch.stack(
                [
                    torch.tensor(
                        [1] * sample["x"]["seq"].shape[-1]
                        + [0] * (max_length - sample["x"]["seq"].shape[-1])
                    )
                    for sample in samples
                ]
            )

        return batch

    @staticmethod
    def get_dhparams(dataset_name, block_size, weighted):
        """returns the data handling parameters for a dataset"""

        dhparams = DD()
        dhparams.FOLDER = dataset_name
        dhparams.TRAIN = os.path.join(
            dhparams.FOLDER, "TRAIN.fasta"
        )  # path relative to input_directory_path
        dhparams.VAL = os.path.join(
            dhparams.FOLDER, "VAL.fasta"
        )  # path relative to input_directory_path
        dhparams.TEST = os.path.join(
            dhparams.FOLDER, "TEST.fasta"
        )  # path relative to input_directory_path
        dhparams.BLOCK_SIZE = block_size
        dhparams.WEIGHTED = weighted
        return dhparams

    @staticmethod
    def raw_to_input(dhparams):
        """converts raw data to input data

        the dhparams must contain the following attributes:
        - RAW_PATH: str - Path to the raw data
        - INPUT_FOLDER: str - Path of folder, where the input data should be stored
        - FILTER: DD - Filter parameters
            - LENGTH: DD - Length filter parameters
                - MIN: int - Minimum length of a sequence
                - MAX: int - Maximum length of a sequence
        - SEED: int - Seed for the random number generator
        """

        # Load sequences from fasta file
        df = read_fasta(dhparams.RAW_PATH, return_df=True, evaluate=False)
        log_info(
            f"Loaded {df.shape[0]} different sequences "
            f"from {dhparams.RAW_PATH} (total {df.cnt.sum()})"
        )

        # Apply filters to sequences
        _filter = []
        if "LENGTH" in dhparams.FILTER:
            if "MIN" in dhparams.FILTER.LENGTH:
                _filter.append(f"{dhparams.FILTER.LENGTH.MIN} < length")
            if "MAX" in dhparams.FILTER.LENGTH:
                _filter.append(f"length < {dhparams.FILTER.LENGTH.MAX}")
        _filter = " and ".join(_filter)
        df = df.query(_filter)
        log_info(f"Applied {_filter}")
        log_info(f"  and obtained {df.shape[0]} sequences")

        # split dataframe into TRAIN, VAL, TEST, PREDICT
        np.random.seed(dhparams.SEED)
        df = df.iloc[np.random.permutation(range(df.shape[0]))]
        von, bis = 0, 0
        for split in Split:
            von = bis
            pc = dhparams.SPLIT[str(split)]
            if pc is not None:
                if pc == "*":
                    df_tmp = df.iloc[von:]
                else:
                    bis += int(df.shape[0] * pc)
                    df_tmp = df.iloc[von:bis]
                log_info(
                    f"Write {df_tmp.shape[0]} sequences to {str(split)} file ({pc})"
                )
                write_fasta(join(dhparams.INPUT_FOLDER, f"{str(split)}.fasta"), df_tmp)
