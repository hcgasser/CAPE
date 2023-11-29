import os

from kit.log import log_info
from kit.bioinf.fasta import fastas_to_seqs
from kit.bioinf.pdb import pdb_to_seqs

from .utils import get_seq_hash
from .path import get_fasta_file_path, get_pdb_file_path


def check_seq_hashes_in_loch(
    loch_path=None, predictor_structure_name="AF", repair_filename=False
):
    fasta_mismatches, pdb_mismatches = [], []
    checked = 0
    for dir_path, _, file_names in os.walk(get_fasta_file_path(loch_path=loch_path)):
        for file_name in file_names:
            if file_name.endswith(".fasta"):
                fasta_file_path = os.path.join(dir_path, file_name)
                filename_seq_hash = file_name.split(".")[0]

                seq = fastas_to_seqs(fasta_file_path)[0]
                seq_hash = get_seq_hash(seq)

                if filename_seq_hash != seq_hash:
                    fasta_mismatches.append((filename_seq_hash, seq_hash))
                    if repair_filename:
                        os.rename(
                            fasta_file_path, os.path.join(dir_path, f"{seq_hash}.fasta")
                        )
                checked += 1
    log_info(f"Checked {checked} fasta files - {len(fasta_mismatches)} mismatches")

    checked = 0
    for dir_path, _, file_names in os.walk(
        get_pdb_file_path(
            loch_path=loch_path, predictor_structure_name=predictor_structure_name
        )
    ):
        for file_name in file_names:
            if file_name.endswith(".pdb"):
                pdb_file_path = os.path.join(dir_path, file_name)
                filename_seq_hash = file_name.split("_")[0]

                seq = list(pdb_to_seqs(pdb_file_path)[0].values())[0]
                seq_hash = get_seq_hash(seq)

                if filename_seq_hash != seq_hash:
                    pdb_mismatches.append((filename_seq_hash, seq_hash))
                    if repair_filename:
                        os.rename(
                            pdb_file_path,
                            os.path.join(
                                dir_path, f"{seq_hash}_{predictor_structure_name}.pdb"
                            ),
                        )
                checked += 1
    log_info(f"Checked {checked} pdb files - {len(pdb_mismatches)} mismatches")

    return fasta_mismatches, pdb_mismatches
