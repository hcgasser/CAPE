import random
import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from kit.log import log_info, log_caller
from kit.globals import ENV
from kit.path import join
from kit.bioinf import seq_to_idx, aa_1_to_idx
from kit.bioinf.fasta import read_fasta, write_fasta
from kit.data import Split


class FastaDS(Dataset):
    def __init__(self, split, DHPARAMS, mask=False):
        super().__init__()
        log_caller(self)

        if str(split) in DHPARAMS:
            input_file = os.path.join(ENV.INPUT, DHPARAMS[str(split)])
            if os.path.exists(input_file):
                self.df = read_fasta(input_file, return_df=True, evaluate=True)
                self.df['weight'] = self.df.shape[0] * self.df.cnt / self.df.cnt.sum()
                log_info(f"Loaded {self.df.shape[0]} samples for the {split} set")
            else:
                log_info(f"File {input_file} does not exist")
                self.df = None
        else:
            self.df = None

        self.block_size = DHPARAMS.BLOCK_SIZE if 'BLOCK_SIZE' in DHPARAMS else None
        self.weighted = DHPARAMS.WEIGHTED
        self.mask = mask

        log_info(f"block_size: {self.block_size} ")
        log_info(f"weighted: {self.weighted} ")

    def __len__(self):
        return len(self.df) if self.df is not None else 0

    def __getitem__(self, j):
        row = self.df.iloc[j]
        seq = "*" + str(row.name)
        weight = row.weight if self.weighted else 1.
        # seq, weight = random.choice([("*ACDEFGHIKACDEFGHIK*", 7), ("*ACDEFGHIKAAAAACDEFGHIK*", 2), ("*ADEFGHIKAAAAACDEFGHIK*", 1)])
        if self.block_size is not None and self.block_size < len(seq):
            start = random.randint(0, len(seq) - self.block_size - 1)
            seq = seq[start:(start + self.block_size + 1)]  # a returned sequence must have block_size + 1 elements (the last is only a target)

        result = {
            'x': {
                'seq': torch.tensor(seq_to_idx(seq)),
                'weight': torch.tensor(weight)
            }
        }
        return result

    def collate(self, samples):
        max_length = np.max([sample['x']['seq'].shape[-1] for sample in samples])

        batch = {
            'x': {
                'seq': torch.stack([
                    F.pad(
                        sample['x']['seq'],
                        (0, max_length - sample['x']['seq'].shape[-1]),
                        mode='constant', value=aa_1_to_idx['-']
                    ) for sample in samples], dim=0),
                'weight': torch.stack([
                    sample['x']['weight'] for sample in samples]),
            }
        }
        if self.mask:
            batch['x']['mask'] = torch.stack([
                torch.tensor([1] * sample['x']['seq'].shape[-1]
                             + [0]*(max_length - sample['x']['seq'].shape[-1]))
                for sample in samples])

        return batch

    @staticmethod
    def raw_to_input(DHPARAMS):
        # Load sequences from fasta file
        df = read_fasta(DHPARAMS.RAW_PATH, return_df=True, evaluate=False)
        log_info(f"Loaded {df.shape[0]} different sequences from {DHPARAMS.RAW_PATH} (total {df.cnt.sum()})")

        # Apply filters to sequences
        filter = []
        if "LENGTH" in DHPARAMS.FILTER:
            if "MIN" in DHPARAMS.FILTER.LENGTH:
                filter.append(f"{DHPARAMS.FILTER.LENGTH.MIN} < length")
            if "MAX" in DHPARAMS.FILTER.LENGTH:
                filter.append(f"length < {DHPARAMS.FILTER.LENGTH.MAX}")
        filter = " and ".join(filter)
        df = df.query(filter)
        log_info(f"Applied {filter}")
        log_info(f"  and obtained {df.shape[0]} sequences")

        # split dataframe into TRAIN, VAL, TEST, PREDICT
        np.random.seed(DHPARAMS.SEED)
        df = df.iloc[np.random.permutation(range(df.shape[0]))]
        von, bis = 0, 0
        for split in Split:
            von = bis
            pc = DHPARAMS.SPLIT[str(split)]
            if pc is not None:
                if pc == '*':
                    df_tmp = df.iloc[von:]
                else:
                    bis += int(df.shape[0] * pc)
                    df_tmp = df.iloc[von:bis]
                log_info(f"Write {df_tmp.shape[0]} sequences to {str(split)} file ({pc})")
                write_fasta(join(DHPARAMS.INPUT_FOLDER, f"{str(split)}.fasta"), df_tmp)
