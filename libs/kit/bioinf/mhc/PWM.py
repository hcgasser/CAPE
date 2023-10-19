import os
import numpy as np
import pandas as pd

import kit.globals as G
from kit.bioinf import aa1_std
from kit.bioinf.mhc import to_HLA_prefix, MHC1Predictor


class PWMPredictor(MHC1Predictor):
    def __init__(self, folder=None, limit=0.02, aa_na=0):
        super().__init__("PWM", None, limit)
        self.folder = folder if folder is not None else os.path.join(G.ENV.INPUT, "pwm")
        self.factor = 0.01
        self.PWMs_log = {}
        self.info_pc = {}
        self.limits = {}
        self.aa_na = aa_na
        self.aa_idx = None

    def save(self):
        pass

    def predict_peptides(self, peptides, allele):
        for peptide in peptides:
            self.predict_peptide(peptide, allele)

    def load_allele(self, allele, length):
        if allele not in self.PWMs_log:
            self.PWMs_log[allele] = {}
            self.info_pc[allele] = {}
            self.limits[allele] = {}

        if length not in self.PWMs_log[allele]:
            allele_for_path = allele.replace('*', '_')
            self.PWMs_log[allele][length] = pd.read_csv(
                os.path.join(self.folder, allele_for_path, f"{allele_for_path}-{length}_log.csv")
            ).set_index('AA').sort_index()
            self.PWMs_log[allele][length].columns = [int(c) for c in self.PWMs_log[allele][length].columns]
            if self.aa_idx is None:
                self.aa_idx = {}
                for idx, aa in enumerate(self.PWMs_log[allele][length].index):
                    self.aa_idx[aa] = idx
            else:
                for idx, aa in enumerate(self.PWMs_log[allele][length].index):
                    assert self.aa_idx[aa] == idx

            info = pd.read_csv(os.path.join(self.folder, allele_for_path,
                                            f"pc-{allele.replace('*', '_')}-{self.limit}-{length}.csv"
                                            )).set_index('Info')
            info = info.loc[info.index.str.startswith('pc_')]
            info['pc'] = info.apply(lambda row: float(row.name[3:]), axis=1)
            info = info[['pc', allele]].set_index('pc').sort_index(ascending=True)
            self.info_pc[allele][length] = info
            self.set_limit_allele_length(allele, length)
            self.PWMs_log[allele][length] = self.PWMs_log[allele][length].values

    def set_limit(self, limit):
        self.limit = limit
        for allele, lengths in self.info_pc.items():
            for length in lengths.keys():
                self.set_limit_allele_length(allele, length)

    def set_limit_allele_length(self, allele, length):
        y = [-2e9] + list(self.info_pc[allele][length][allele].to_numpy()) + [0.]
        x = [0.] + list(self.info_pc[allele][length].index) + [100.]
        self.limits[allele][length] = np.interp(100. - 100. * self.limit, x, y)

    def peptide_presented(self, peptide, allele, add_if_missing=True, predict_if_missing=False):
        length = len(peptide)
        score = self.score_peptide(peptide, allele)
        return score > self.limits[allele][length]

    def predict_peptide(self, peptide, allele):
        length = len(peptide)
        self.load_allele(allele, length)

        score = self.score_peptide(peptide, allele)

        x = [-2e9] + list(self.info_pc[allele][length][allele].to_numpy()) + [0.]
        y = [0.] + list(self.info_pc[allele][length].index) + [100.]

        self.updated_percentiles = True
        self.percentiles[allele][peptide] = 100 - np.interp(score, x, y)

    def score_peptide(self, peptide, allele):
        length = len(peptide)
        score = 0.
        for pos, aa in enumerate(peptide):
            if aa in self.aa_idx:
                # score += self.PWMs_log[allele][length].loc[aa, pos]
                score += self.PWMs_log[allele][length][self.aa_idx[aa], pos]
            else:
                score += self.aa_na
        return score
