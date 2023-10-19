import os
import re
import subprocess
import tempfile

import pdb
import pandas as pd
from tqdm.auto import tqdm

import kit.path
from kit.data import str_to_file
from kit.bioinf.mhc import get_HLA_type, get_HLA_split, MHC1Predictor
from kit.utils import temp_working_directory


class netMHCpanPredictor(MHC1Predictor):
    def __init__(self, folder=None, limit=0.02, MHC_I_alleles_to_load=None):
        super().__init__("netMHCpan", folder, limit, MHC_I_alleles_to_load=MHC_I_alleles_to_load)
        self.factor = 0.01

    def predict_peptides(self, peptides, allele):
        with temp_working_directory() as tmp_dir_path:
            str_to_file("\n".join(peptides), f"temporary.pep")
            result = subprocess.run(["netMHCpan", f"-p temporary.pep", f"-a {to_HLA_netMHCpan(allele)}"],
                                    capture_output=True)
            interpret_netMHCpan_output(result.stdout.decode('UTF-8').split("\n"), self.percentiles)
            self.updated_percentiles = True


def interpret_netMHCpan_output(lines, el_ranks):
    line_nr = 1
    allele = None
    for line in lines:
        if line_nr == 42:
            allele = f"{line[:5]}*{line[5:10]}"

        if line_nr >= 49 and line[5:8] == "HLA":
            regex = f"\s+\d+\s+HLA-[ABCEFGKL]\*\d+:\d+\s+(\w+)\s+(?:\w|-)+\s+\d+\s+\d+\s+\d+\s+\d+\s+\d+\s+\w+\s+PEPLIST\s+([\d.]+)\s+([\d.]+)"
            peptide, score, rank_el = re.findall(regex, line)[0]
            if peptide.isnumeric():
                pdb.set_trace()
            el_ranks[allele][peptide] = float(rank_el)

        line_nr += 1


def to_HLA_netMHCpan(mhc):
    if get_HLA_type(mhc) == 'MHC-I':
        h = get_HLA_split(mhc)
        return f"HLA-{h[0]}{h[1]}:{h[2]}"