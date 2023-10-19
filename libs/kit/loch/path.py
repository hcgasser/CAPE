import os

from kit.path import join


LOCH_PATH = os.environ["LOCH"]


def get_pdb_file_path(seq_hash=None, loch_path=None, predictor_structure_name='AF'):
    loch_path = loch_path if loch_path is not None else LOCH_PATH
    if seq_hash is None:
        return join(loch_path, 'structures', predictor_structure_name, 'pdb')
    return join(loch_path, 'structures', predictor_structure_name, 'pdb', f"{seq_hash}_{predictor_structure_name}.pdb")


def get_fasta_file_path(seq_hash=None, loch_path=None):
    loch_path = loch_path if loch_path is not None else LOCH_PATH
    if seq_hash is None:
        return join(loch_path, 'sequences')
    return join(loch_path, 'sequences', f"{seq_hash}.fasta")


def get_function_path(seq_hash=None, loch_path=None, predictor_structure_name='AF', predictor_function_name='TransFun'):
    """ Get the path to the MD files for a given sequence hash and MD parameter hash

    :param seq_hash: The sequence hash
    :param md_param_hash: The MD parameter hash
    :param loch_path: The path to the protein files
    :param predictor_structure_name: The name of the structure predictor
    :param predictor_function_name: The name of the function predictor
    :return: The path to the MD files
        if seq_hash and md_param_hash are None: The path to the MD files directory
        else: the path to the directory with the MD files for the given sequence hash and MD parameter hash
    """

    loch_path = loch_path if loch_path is not None else LOCH_PATH
    if seq_hash is None:
        return join(loch_path, 'function', 'GO', predictor_function_name)
    return join(loch_path, 'function', 'GO', predictor_function_name, f"{seq_hash}_{predictor_structure_name}.txt")


def get_md_path(seq_hash, md_param_hash=None, loch_path=None, predictor_structure_name='AF'):
    loch_path = loch_path if loch_path is not None else LOCH_PATH
    if seq_hash is None and md_param_hash is None:
        return join(loch_path, 'dynamics')
    elif md_param_hash is None:
        return os.path.join(loch_path, 'dynamics', f"{seq_hash}_{predictor_structure_name}")
    return os.path.join(loch_path, 'dynamics', f"{seq_hash}_{predictor_structure_name}", md_param_hash)