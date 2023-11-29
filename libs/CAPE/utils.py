""" some auxiliary functions for CAPE """

import os
import hashlib

from kit.loch import file_to_seq_hashes
from kit.bioinf.fasta import translate_fasta


def get_seq_set_hash(seq_hashes):
    """Returns the SHA-256 hash of a set of sequence hashes.
    This is useful to compare two sets of sequences for equality."""

    seq_hashes = "\n".join(sorted(seq_hashes))

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string as bytes and update the hash object
    hash_object.update(seq_hashes.encode("utf-8"))

    # Get the hexadecimal representation of the hash digest
    hash_code = hash_object.hexdigest()

    return hash_code


def cp_translated_fastas_to_path(
    seq_hashes_file_path, input_files_path, output_files_path, translate=("", "", "*-")
):
    """copies a set of translated fasta files to a new path"""

    for seq_hash in file_to_seq_hashes(seq_hashes_file_path):
        translate_fasta(
            os.path.join(input_files_path, seq_hash.strip() + ".fasta"),
            os.path.join(output_files_path, seq_hash.strip() + ".fasta"),
            translate=translate,
        )
