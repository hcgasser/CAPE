""" This module contains functions for aligning two structures using the CE algorithm. """


import os

from Bio.PDB import PDBParser
from Bio.PDB.cealign import CEAligner


def align_structures(pdb1_file_path, pdb2_file_path):
    """Aligns two structures using the CE algorithm.

    :param pdb1_file_path: str - path to the first structure
    :param pdb2_file_path: str - path to the second structure
    :return: float - RMSD of the alignment
    """

    if not os.path.exists(pdb1_file_path):
        raise Exception(f"CEAligner input file {pdb1_file_path} does not exist")
    if not os.path.exists(pdb2_file_path):
        raise Exception(f"CEAligner input file {pdb2_file_path} does not exist")

    aligner = CEAligner()
    aligner.set_reference(PDBParser().get_structure("reference", pdb1_file_path))
    aligner.align(PDBParser().get_structure("reference", pdb2_file_path))
    return aligner.rms
