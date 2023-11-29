"""MHC1 predictor based on PWMs."""

import os
import numpy as np
import pandas as pd

from kit.bioinf.immuno.mhc_1 import Mhc1Predictor


class Mhc1PredictorPwm(Mhc1Predictor):
    """Fast MHC1 predictor based on PWMs.

    :param data_dir_path: Path to the folder with the PWMs.
    :param limit: Percentile limit below which a peptide is considered presented.
    :param aa_na: Score for amino acids not present in the PWM.
        The score of a peptide is the sum of the scores of its amino acids.
        Each score is the log likelihood of the amino acid at the given position
            in presented peptides of the allele.
        The percentile of a peptide is the percentile of its score among all peptides
            This is approximated via linear interpolation
    """

    def __init__(self, data_dir_path=None, limit=0.02, aa_na=0):
        super().__init__("PWM", data_dir_path, limit)
        self.factor = 0.01
        self.PWMs_log = {}
        self.info_pc = {}
        self.score_limits = {}
        self.aa_na = aa_na
        self.aa_idx = None

    def save(self):
        """no information should be saved for this predictor"""
        pass

    def predict_peptides(self, peptides, allele):
        """Predicts the percentiles of the peptides for the allele."""
        for peptide in peptides:
            self.predict_peptide(peptide, allele)

    def load_allele(self, allele, length):
        """Loads the PWMs for the allele and length.

        :param allele: Allele name.
        :param length: Peptide length.
        """

        if allele not in self.PWMs_log:
            self.PWMs_log[allele] = {}
            self.info_pc[allele] = {}
            self.score_limits[allele] = {}

        if length not in self.PWMs_log[allele]:
            allele_for_path = allele.replace("*", "_")
            self.PWMs_log[allele][length] = (
                pd.read_csv(
                    os.path.join(
                        self.data_dir_path,
                        allele_for_path,
                        f"{allele_for_path}-{length}_log.csv",
                    )
                )
                .set_index("AA")
                .sort_index()
            )
            self.PWMs_log[allele][length].columns = [
                int(c) for c in self.PWMs_log[allele][length].columns
            ]
            if self.aa_idx is None:
                self.aa_idx = {}
                for idx, aa in enumerate(self.PWMs_log[allele][length].index):
                    self.aa_idx[aa] = idx
            else:
                for idx, aa in enumerate(self.PWMs_log[allele][length].index):
                    assert self.aa_idx[aa] == idx

            info = pd.read_csv(
                os.path.join(
                    self.data_dir_path,
                    allele_for_path,
                    f"pc-{allele.replace('*', '_')}-{self.limit}-{length}.csv",
                )
            ).set_index("Info")
            info = info.loc[info.index.str.startswith("pc_")]
            info["pc"] = info.apply(lambda row: float(row.name[3:]), axis=1)
            info = info[["pc", allele]].set_index("pc").sort_index(ascending=True)
            self.info_pc[allele][length] = info
            self.set_score_limit_allele_length(allele, length)
            self.PWMs_log[allele][length] = self.PWMs_log[allele][length].values

    def set_score_limit_allele_length(self, allele, length):
        """Sets the score limit for the allele and length. If the score of a peptide
        is above this score limit, it is considered presented. (meaning its percentile
        is lower than the limit)"""

        y = [-2e9] + list(self.info_pc[allele][length][allele].to_numpy()) + [0.0]
        x = [0.0] + list(self.info_pc[allele][length].index) + [100.0]
        self.score_limits[allele][length] = np.interp(100.0 - 100.0 * self.limit, x, y)

    def peptide_presented(
        self, peptide, allele, queue_if_missing=True, predict_if_missing=False
    ):
        """Returns whether the peptide is predicted to be presented by the allele."""

        length = len(peptide)
        score = self.score_peptide(peptide, allele)
        return score > self.score_limits[allele][length]

    def predict_peptide(self, peptide, allele):
        """Predicts the percentile of the peptide for the allele."""

        length = len(peptide)
        self.load_allele(allele, length)

        score = self.score_peptide(peptide, allele)

        x = [-2e9] + list(self.info_pc[allele][length][allele].to_numpy()) + [0.0]
        y = [0.0] + list(self.info_pc[allele][length].index) + [100.0]

        self.updated_percentiles = True
        self.percentiles[allele][peptide] = 100 - np.interp(score, x, y)

    def score_peptide(self, peptide, allele):
        """Returns the score of the peptide for the allele."""

        length = len(peptide)
        score = 0.0
        for pos, aa in enumerate(peptide):
            if aa in self.aa_idx:
                # score += self.PWMs_log[allele][length].loc[aa, pos]
                score += self.PWMs_log[allele][length][self.aa_idx[aa], pos]
            else:
                score += self.aa_na
        return score
