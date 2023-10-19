import os
import hashlib

from kit.log import log_info
from kit.loch import file_to_seq_hashes
from kit.loch.utils import get_seq_hash
from kit.bioinf.fasta import translate_fasta, fastas_to_seqs
from kit.bioinf.pdb import pdb_to_seqs


def get_rep_base_hash(seq_hashes):
    seq_hashes = "\n".join(sorted(seq_hashes))

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string as bytes and update the hash object
    hash_object.update(seq_hashes.encode('utf-8'))

    # Get the hexadecimal representation of the hash digest
    hash_code = hash_object.hexdigest()

    return hash_code


def cp_translated_fastas_to_path(seq_hashes_file_path, input_files_path, output_files_path, translate=('', '', '*-')):
    for seq_hash in file_to_seq_hashes(seq_hashes_file_path):
        translate_fasta(os.path.join(input_files_path, seq_hash.strip() + '.fasta'), os.path.join(output_files_path, seq_hash.strip() + '.fasta'), translate=translate)


