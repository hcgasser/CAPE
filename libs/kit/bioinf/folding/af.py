"""Module to convert FASTA files to PDB files with AlphaFold."""

import os
import subprocess
from kit.log import log_info
from kit.path import get_entries
from kit.bioinf.fasta import translate_fasta


def fasta_to_pdb(path, regex):
    """Converts FASTA files to PDB files with AlphaFold."""

    fasta_files = get_entries(path, regex, returndict=False)
    for fasta_file in fasta_files:
        filename = fasta_file.split(os.sep)
        dirname = os.sep.join(filename[:-1])
        parent_folder = filename[-2]
        filename = filename[-1]
        filename_root = filename.removesuffix(".fasta")
        if (
            not os.path.exists(
                os.path.join(dirname, filename_root, filename_root + "_AF.pdb")
            )
            and parent_folder != filename_root
        ):
            log_info(f"converting {filename} to PDB with AlphaFold")
            translate_fasta(fasta_file)
            command = ["af.sh", fasta_file]
            subprocess.call(command)
