import hashlib


def get_seq_hash(seq, translate=('', '', '*-')):

    seq = seq.translate(str.maketrans(*translate))

    # Create a SHA-256 hash object
    hash_object = hashlib.sha256()

    # Encode the string as bytes and update the hash object
    hash_object.update(seq.encode('utf-8'))

    # Get the hexadecimal representation of the hash digest
    hash_code = hash_object.hexdigest()

    return hash_code