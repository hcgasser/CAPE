import os
import shutil
from tqdm.auto import tqdm

from kit.data import str_to_file
from kit.log import log_info
from kit.path import join


predictors_structure = ['AF', 'ESM']
predictors_go_function = ['TransFun']


def file_to_seq_hashes(file_path):
    if not os.path.exists(file_path):
        raise Exception(f"{file_path} does not exist")
    
    with open(file_path, 'r') as f:
        seq_hashes = f.read()

    return [seq_hash for seq_hash in seq_hashes.split('\n') if seq_hash != '']


def seq_hashes_to_file(seq_hashes, file_path, append=False):
    text = "".join([f"{seq_hash}\n" for seq_hash in seq_hashes])
    str_to_file(text, file_path, append)


def copy_info(seq_hash_file_path, source_loch_path, target_loch_path, cp_sequence=True, cp_structure=True, cp_dynamics=True, cp_function=True):
    seq_hashes = file_to_seq_hashes(seq_hash_file_path)

    c_fastas, c_pdb, c_dynamics, c_function = 0, 0, 0, 0
    for seq_hash in tqdm(seq_hashes):
        if cp_sequence:
            source_file_path = os.path.join(source_loch_path, 'sequences', f"{seq_hash}.fasta")
            if os.path.exists(source_file_path):
                shutil.copyfile(source_file_path, join(target_loch_path, 'sequences', f"{seq_hash}.fasta"))
                c_fastas += 1

        if cp_structure:
            for predictor_structure in predictors_structure:
                source_file_path = os.path.join(source_loch_path, 'structures', predictor_structure, 'pdb', f"{seq_hash}_{predictor_structure}.pdb")
                if os.path.exists(source_file_path):
                    shutil.copyfile(source_file_path, join(target_loch_path, 'structures', predictor_structure, 'pdb', f"{seq_hash}_{predictor_structure}.pdb"))
                    c_pdb += 1

                source_file_path = os.path.join(source_loch_path, 'structures', predictor_structure, 'pkl', f"{seq_hash}.pkl")
                if os.path.exists(source_file_path):
                    shutil.copyfile(source_file_path, join(target_loch_path, 'structures', predictor_structure, 'pkl', f"{seq_hash}.pkl"))

        if cp_dynamics:
            for predictor_structure in predictors_structure:
                source_dir_path = os.path.join(source_loch_path, 'dynamics', f"{seq_hash}_{predictor_structure}")
                if os.path.exists(source_dir_path):
                    shutil.copytree(source_dir_path, join(target_loch_path, 'dynamics', f"{seq_hash}_{predictor_structure}"))
                    c_dynamics += 1

        if cp_function:
            for predictor_structure in predictors_structure:
                for predictor_go_function in predictors_go_function:
                    source_file_path = os.path.join(source_loch_path, 'function', 'GO', predictor_go_function, f"{seq_hash}_{predictor_structure}.txt")
                    if os.path.exists(source_file_path):
                        shutil.copyfile(source_file_path, join(target_loch_path, 'function', 'GO', predictor_go_function, f"{seq_hash}_{predictor_structure}.txt"))
                        c_function += 1

    log_info(f"Copied {c_fastas} fastas, {c_pdb} pdbs, {c_dynamics} dynamics, {c_function} functions")
