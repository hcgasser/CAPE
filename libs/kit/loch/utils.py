""" Utility functions for the loch module. """

import hashlib


def get_seq_hash(seq, translate=("", "", "*-")):
    """Returns the SHA-256 hash of a sequence.

    :param seq: str - sequence
    :param translate: tuple - characters to translate
        the sequence with before its hash is computed
    :return hash_code: str - SHA-256 hash of the sequence
    """

    seq = seq.translate(str.maketrans(*translate))

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string as bytes and update the hash object
    hash_object.update(seq.encode("utf-8"))

    # Get the hexadecimal representation of the hash digest
    hash_code = hash_object.hexdigest()

    return hash_code
