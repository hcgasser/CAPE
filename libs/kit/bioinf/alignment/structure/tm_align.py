""" This module contains functions for aligning two structures 
using the TMalign algorithm. """

import os
import re
import subprocess


def align_structures(pdb_ref, pdb, output=None):
    """aligns two structures using the TMalign algorithm.

    :param pdb_ref: str - path to the reference structure
    :param pdb: str - path to the structure to be aligned
    :param output: str - path to the output file (Optional)
    :return: tuple - (TM-score, aligned_length, RMSD, identical)
    """

    if not os.path.exists(pdb_ref):
        raise Exception(f"TMalign input file {pdb_ref} does not exist")
    if not os.path.exists(pdb):
        raise Exception(f"TMalign input file {pdb} does not exist")

    if output is not None:
        command = ["TMalign", pdb, pdb_ref, "-o", output]
    else:
        command = ["TMalign", pdb, pdb_ref]

    result = subprocess.run(command, capture_output=True, check=False)
    res = result.stdout.decode("utf-8")

    tm_score, aligned_length, rmsd, identical = None, None, None, None
    for line in res.split("\n"):
        regres = re.findall(
            r"^TM-score=\s+(\d\.\d+)\s\(if normalized by length of Chain_2,", line
        )
        if len(regres) == 1 and tm_score is None:
            tm_score = float(regres[0])

        regres = re.findall(
            r"Aligned length=\s+(\d+), RMSD=\s+(\d+\.\d+), "
            + r"Seq_ID=n_identical/n_aligned=\s+(\d+\.\d+)",
            line,
        )
        if len(regres) == 1:
            aligned_length = int(regres[0][0])
            rmsd = float(regres[0][1])
            identical = float(regres[0][2])

    return tm_score, aligned_length, rmsd, identical
