"""This module specifies the base class for MHC Class 1 binding predictors"""

import os
import re
from collections import defaultdict
import importlib

from tqdm.auto import tqdm

import numpy as np
import pandas as pd

from kit.path import join
from kit.loch.utils import get_seq_hash
from kit.bioinf import get_kmers, seq_to_kmers_list
from kit.log import log_info, log_warning

from kit.bioinf.immuno.utils import get_mhc_name_split


MHC_1_PEPTIDE_LENGTHS = [8, 9, 10]
MHC_2_pep_lengths = [13, 14, 15, 16, 17]


class Mhc1Predictor:
    """Base class for MHC Class 1 binding predictors

    :param name: str - name of the predictor
    :param data_dir_path: str - path to the directory with the predictor data
        this can include already predicted peptides and percentiles
    :param limit: float - percentile limit for the predictor (e.g. 0.02)
        values below this limit are considered to be presented
    :param mhc_1_alleles_to_load: list - list of alleles to load from the data_dir_path
    """

    def __init__(self, name, data_dir_path, limit, mhc_1_alleles_to_load=None):
        log_info(f"Init {type(self)} MHC-Class1 predictor")
        self.percentiles = defaultdict(
            lambda: {}
        )  # 2 level dictionary: el_ranks[mhc][peptide] = el_rank
        self.missing = defaultdict(
            lambda: []
        )  # collect the missing peptides to be requested from the predictor in one go
        self.unavailable = defaultdict(
            lambda: set()
        )  # collect the peptides that could not be predicted
        self.factor = 1
        self.data_dir_path = data_dir_path
        self.name = name
        if data_dir_path is not None:
            self.load_percentiles(
                data_dir_path, mhc_1_alleles_to_load=mhc_1_alleles_to_load
            )
        self.limit = limit
        self.updated_percentiles = False

    @staticmethod
    def get_predictor(class_name):
        """Returns the predictor class specified by the class_name

        Loads the module with the predictor (its name is equal to '_' plus the class name
        without the Mhc1Predictor prefix).
        From this module it returns the class that has the class_name

        :param class_name: str - name of the predictor class
        :return: class - the predictor class
        """

        name_module = class_name.removeprefix("Mhc1Predictor").lower()
        # pylint: disable=unused-variable
        module = importlib.import_module(
            f"._{name_module}", package="kit.bioinf.immuno.mhc_1"
        )
        return eval(f"module.{class_name}")

    def save(self):
        """Saves the percentiles to the data_dir_path is any changes were made"""

        if self.updated_percentiles:
            self.save_percentiles(self.data_dir_path)

    def load_percentiles(
        self, data_dir_path, prefix="percentile", mhc_1_alleles_to_load=None
    ):
        """Loads the percentiles from the data_dir_path"""

        for dirname, _, filenames in os.walk(data_dir_path, followlinks=True):
            for filename in filenames:
                h = re.findall(
                    rf"^{prefix}_HLA_([ABCEFGKL])_(\d+)_(\d+)\.tsv$", filename
                )
                if len(h) == 1 and len(h[0]) == 3:
                    mhc = f"HLA-{h[0][0]}*{h[0][1]}:{h[0][2]}"
                    if mhc_1_alleles_to_load is None or mhc in mhc_1_alleles_to_load:
                        log_info(f"Loading {mhc} from {filename}")
                        with open(os.path.join(dirname, filename), "r") as f:
                            perc_file = f.read()
                        perc_file = perc_file.split("\n")
                        for line in perc_file[1:]:
                            line = line.split("\t")
                            if len(line) == 2:
                                peptide, percentile = line
                                self.percentiles[mhc][peptide] = float(percentile)

    def save_percentiles(self, data_dir_path, prefix="percentile"):
        """Saves the percentiles to the data_dir_path"""

        for mhc, mhc_percentiles in self.percentiles.items():
            hla_split = get_mhc_name_split(mhc)
            filename = "_".join([prefix, "HLA"] + list(hla_split))
            filename = join(data_dir_path, f"{filename}.tsv")

            df = pd.DataFrame(
                index=list(mhc_percentiles.keys()),
                columns=[prefix],
                data=list(mhc_percentiles.values()),
            )
            df.index.name = "peptide"
            df.to_csv(filename, sep="\t")

    def peptide_presented(
        self, peptide, allele, queue_if_missing=True, predict_if_missing=False
    ):
        """Predicts if a peptide is presented by the specified MHC allele

        For this is compares the predicted rank of the peptide to the limit.
        The peptide is considered to be presented by the allele if the rank is
        below the limit.

        Can also return None if the pMHC has not been predicted yet.

        :param peptide: str - peptide sequence
        :param allele: str - MHC allele name
        :param queue_if_missing: bool - should the peptide be added to the missing peptides
            if it has not been predicted yet. Defaults to True.
            This will cause the peptide to be predicted in the next call of
            predict_missing_peptides.
        :param predict_if_missing: bool - should the peptide be predicted if it has not
            been predicted yet. Defaults to False.
            This will cause not to wait for the next call of predict_missing_peptides,
            but predict the peptide immediately.
        :return: bool - True if the peptide is presented by the allele, False if not
            or None if the peptide has not been predicted yet (and predict_if_missing
            is False)
        """

        result = self.peptide_rank(
            peptide, allele, queue_if_missing, predict_if_missing
        )
        return None if (result is None) else (result < self.limit)

    def peptide_rank(
        self, peptide, allele, queue_if_missing=True, predict_if_missing=False
    ):
        """Returns the predicted rank of a peptide for the specified MHC allele

        Args: see peptide_presented
        """
        if peptide in self.percentiles[allele]:
            return self.percentiles[allele][peptide] * self.factor

        if predict_if_missing:
            self.predict_peptides(peptides=[peptide], allele=allele)
            return self.peptide_rank(
                peptide, allele, queue_if_missing=False, predict_if_missing=False
            )

        if queue_if_missing:
            self.missing[allele].append(peptide)
        return None

    def predict_peptides(self, peptides, allele):
        """Calls the underlying prediction algorithm
        and saves the results in the self.percentiles attribute
        Has to be implemented by the child class

        :param peptides: list - list of peptide sequences
        :param allele: str - MHC allele name
        """
        raise NotImplementedError()

    def seq_presented(self, seq, alleles, lengths=None):
        """finds all peptides within a sequence that get presented by
        the specified MHC alleles

        :param seq: str - protein sequence
        :param alleles: list - list of MHC allele names
        :param lengths: list - list of peptide lengths to consider
            Defaults to MHC_1_PEPTIDE_LENGTHS
        :return: list - list of tuples (peptide, allele, rank,
            last position of peptide in the sequence)
        """
        if lengths is None:
            lengths = MHC_1_PEPTIDE_LENGTHS

        presented = []
        seq = seq.replace("*", "").replace("-", "").replace("?", "")
        if isinstance(alleles, str):
            alleles = [alleles]

        if len(seq) >= min(lengths):
            kmers = get_kmers(seq, lengths)

            for allele in alleles:
                for kmer in kmers:
                    self.peptide_rank(kmer, allele)

                # process all peptides that are not available in the library
                self.predict_missing_peptides()

                for length in lengths:
                    for end in range(length, len(seq)):
                        peptide = seq[end - length : end]
                        rank = self.peptide_rank(peptide, allele)
                        if rank is None:
                            presented.append((peptide, allele, None, end - 1))
                        elif rank < self.limit:
                            presented.append((peptide, allele, rank, end - 1))

        return presented

    def predict_missing_peptides(self):
        """Predicts all peptides that have been queued for prediction"""

        for allele, peptides in self.missing.items():
            self.predict_peptides(peptides=peptides, allele=allele)
        result = self.missing
        self.missing = defaultdict(lambda: [])
        for allele, peptides in result.items():
            unavail = []
            for peptide in peptides:
                if (
                    peptide not in self.percentiles[allele]
                    and peptide not in self.unavailable[allele]
                ):
                    unavail.append(peptide)
            if len(unavail) > 0:
                log_warning(f"Unable to predict {allele}... {', '.join(unavail)}")
            self.unavailable[allele].update(unavail)
        return result

    def get_seq_kmers(self, seq, allele_names, kmer_length):
        """Returns a dataframe with the predicted ranks of all kmers in the sequence

        Each row represents a kmer in the sequence
        There is a column for each allele in allele_names. The value in the column
            represents the rank of the kmer for the allele.
        There are two additional columns:
            visibility - number of alleles for which the kmer is presented
            presented - True if the kmer is presented by at least one allele
        """

        kmers = seq_to_kmers_list(seq, kmer_length)
        df = pd.DataFrame(index=kmers)
        for allele_name in allele_names:
            df[allele_name] = None
            df[allele_name] = df.apply(
                lambda row: self.peptide_rank(
                    row.name,
                    allele_name,
                    queue_if_missing=False,
                    predict_if_missing=True,
                ),
                axis=1,
            )

        df["visibility"] = df.apply(lambda row: np.sum(row < self.limit), axis=1)
        df["presented"] = df.apply(lambda row: row.visibility > 0, axis=1)
        df.seq_hash = get_seq_hash(seq)
        return df
