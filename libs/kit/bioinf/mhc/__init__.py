import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict
from tqdm.auto import tqdm
import importlib

from kit.path import join
from kit.loch.utils import get_seq_hash
from kit.bioinf import get_kmers, seq_to_kmers_list
from kit.log import log_info, log_warning

MHC_I_pep_lengths = [8, 9, 10]
MHC_II_pep_lengths = [13, 14, 15, 16, 17]
hla_regex = r"HLA-([ABC])\*?(\d{1,3}):(\d{1,3})"


def get_predictor(name):
    module = importlib.import_module(f".{name}", package="kit.bioinf.mhc")
    return eval(f"module.{name}Predictor")


def get_HLA_split(mhc):
    return re.findall(r"^(?:HLA-)*(A|B|C|E|F|G|K|L|DRB1|DRB3|DRB4|DRB5|DQB1)\*(\d+):(\d+)[gQNL]*$", mhc)[0]


def get_HLA_type(mhc):
    if mhc == 'None':
        return None
    else:
        h = get_HLA_split(mhc)
        if h[0] in ('A', 'B', 'C', 'E', 'F', 'G', 'K', 'L'):
            return 'MHC-I'
        else:
            return 'MHC-II'


def to_HLA_prefix(mhc):
    if mhc in ('DRBX*NNNN'):
        result = 'None'
    else:
        h = get_HLA_split(mhc)
        result = f"{h[0]}_{h[1]}_{h[2]}"
    return result


def from_HLA_prefix(hla_prefix):
    h = hla_prefix.split('_')
    result = f"HLA-{h[0]}*{h[1]}:{h[2]}"
    return result


def str_to_HLAs(text, min_HLA_gene_cnt=0, max_HLA_gene_cnt=2):
    """ Finds all the HLA names within a string and returns them as a list

    """

    # HLA-[ABC]\d{2}:\d{2}
    candidate_list = [f"HLA-{x[0]}*{int(x[1]):02d}:{int(x[2]):02d}"
                     for x in re.findall(hla_regex, text)]
    if len(candidate_list) != (text.count("HLA-") - text.count("HLA-D")):  # HLA-D are class 2
        candidate_list = []

    if text.count("HLA") > 1:
        if not (min_HLA_gene_cnt <= text.count("HLA-A") <= max_HLA_gene_cnt) \
                or not (min_HLA_gene_cnt <= text.count("HLA-B") <= max_HLA_gene_cnt) \
                or not (min_HLA_gene_cnt <= text.count("HLA-C") <= max_HLA_gene_cnt):
            candidate_list = []

    candidate_list = list(set(candidate_list))
    candidate_list.sort()

    return candidate_list


class MHC1Predictor:
    def __init__(self, name, folder, limit, MHC_I_alleles_to_load=None):
        log_info(f"Init {type(self)} MHC-Class1 predictor")
        self.percentiles = defaultdict(lambda: {})  # 2 level dictionary: el_ranks[mhc][peptide] = el_rank
        self.missing = defaultdict(lambda: [])  # collect the missing peptides to be requested from the predictor in one go
        self.unavailable = defaultdict(lambda: set())  # collect the peptides that could not be predicted
        self.factor = 1
        self.folder = folder
        self.name = name
        if folder is not None:
            self.load_percentiles(folder, MHC_I_alleles_to_load=MHC_I_alleles_to_load)
        self.limit = limit
        self.updated_percentiles = False

    def save(self):
        if self.updated_percentiles:
            self.save_percentiles(self.folder)

    def load_percentiles(self, folder, prefix="percentile", MHC_I_alleles_to_load=None):
        for dirname, _, filenames in os.walk(folder, followlinks=True):
            for filename in filenames:
                h = re.findall(f"^{prefix}_HLA_([ABCEFGKL])_(\d+)_(\d+)\.tsv$", filename)
                if len(h) == 1 and len(h[0]) == 3:
                    mhc = f"HLA-{h[0][0]}*{h[0][1]}:{h[0][2]}"
                    if MHC_I_alleles_to_load is None or mhc in MHC_I_alleles_to_load:
                        log_info(f"Loading {mhc} from {filename}")
                        with open(os.path.join(dirname, filename), 'r') as f:
                            perc_file = f.read()
                        perc_file = perc_file.split("\n")
                        for line in perc_file[1:]:
                            line = line.split("\t")
                            if len(line) == 2:
                                peptide, percentile = line
                                self.percentiles[mhc][peptide] = float(percentile)

    def save_percentiles(self, folder, prefix="percentile"):
        for mhc, mhc_percentiles in self.percentiles.items():
            HLA_split = get_HLA_split(mhc)
            filename = "_".join([prefix, "HLA"] + [e for e in HLA_split])
            filename = join(folder, f"{filename}.tsv")

            df = pd.DataFrame(index=list(mhc_percentiles.keys()), columns=[prefix],
                              data=list(mhc_percentiles.values()))
            df.index.name = "peptide"
            df.to_csv(filename, sep="\t")

    def peptide_presented(self, peptide, allele, add_if_missing=True, predict_if_missing=False):
        result = self.peptide_rank(peptide, allele, add_if_missing, predict_if_missing)
        return None if (result is None) else (result < self.limit)

    def peptide_rank(self, peptide, allele, queue_if_missing=True, predict_if_missing=False):
        if peptide in self.percentiles[allele]:
            return self.percentiles[allele][peptide] * self.factor
        else:
            if predict_if_missing:
                self.predict_peptides(peptides=[peptide], allele=allele)
                return self.peptide_rank(peptide, allele, queue_if_missing=False, predict_if_missing=False)
            elif queue_if_missing:
                self.missing[allele].append(peptide)
            return None

    def update_percentiles(self, seqs, alleles, lengths=[8, 9, 10]):
        kmers = set()

        for seq in seqs:
            kmers = kmers.union(get_kmers(seq, lengths))

        for allele in tqdm(alleles, "Alleles"):
            # check which kmers need to be assessed by netMHCpan
            for kmer in kmers:
                self.peptide_presented(kmer, allele)

            self.predict_missing_peptides()

    def seq_presented(self, seq, alleles, lengths=[8, 9, 10], proteome_tree=None, disregard=None):
        """ finds all peptides within a sequence that get presented by the specified MHC alleles """
        presented = []
        seq = seq.replace("*", "").replace("-", "").replace("?", "")
        if type(alleles) != list:
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
                        peptide = seq[end - length:end]
                        if proteome_tree is None or not proteome_tree.has_kmer(peptide, disregard=disregard):
                            rank = self.peptide_rank(peptide, allele)
                            if rank is None:
                                presented.append((peptide, allele, None, end-1))
                            elif rank < self.limit:
                                presented.append((peptide, allele, rank, end-1))

        return presented

    def predict_missing_peptides(self):
        for allele, peptides in self.missing.items():
            self.predict_peptides(peptides=peptides, allele=allele)
        result = self.missing
        self.missing = defaultdict(lambda: [])
        for allele, peptides in result.items():
            unavail = []
            for peptide in peptides:
                if peptide not in self.percentiles[allele] and peptide not in self.unavailable[allele]:
                    unavail.append(peptide)
            if len(unavail) > 0:
                log_warning(f"Unable to predict {allele}... {', '.join(unavail)}")
            self.unavailable[allele].update(unavail)
        return result

    def predict_peptides(self, peptides, allele):
        """ Calls the underlying prediction algorithm and saves the results in the self.percentiles attribute """
        raise NotImplementedError()
    
    def get_seq_kmers(self, seq, MHCs, kmer_length):
        kmers = seq_to_kmers_list(seq, kmer_length)
        df = pd.DataFrame(index=kmers)
        for mhc in MHCs:
            df[mhc] = None
            df[mhc] = df.apply(lambda row: self.peptide_rank(row.name, mhc, queue_if_missing=False, predict_if_missing=True), axis=1)

        df['visibility'] = df.apply(lambda row: np.sum(row < self.limit), axis=1)
        df['presented'] = df.apply(lambda row: row.visibility > 0, axis=1)
        df.seq_hash = get_seq_hash(seq)
        return df


def check_full(HLAs):
    if len(HLAs) == 6:
        cnt = {'A': 0, 'B': 0, 'C': 0}
        for hla in HLAs:
            cnt[hla[4]] += 1

        if cnt['A'] == cnt['B'] == cnt['C'] == 2:
            return True

    return False
