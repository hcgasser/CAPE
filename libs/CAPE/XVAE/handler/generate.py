import kit
import kit.globals as G
from kit.loch import seq_hashes_to_file
from kit.loch.utils import get_seq_hash
from kit.loch.path import get_fasta_file_path
from kit.bioinf import unpredicted_tokens
from kit.bioinf.fasta import seqs_to_fasta
from kit.path import join

from CAPE.XVAE import load_model


def run():
    load_model(G.TASK.MODEL_ID)
    if G.TASK.OUTPUT_FOLDER_PATH is not None:
        output_folder_path = G.TASK.OUTPUT_FOLDER_PATH
    else:
        output_folder_path = join(G.TASK.PATHS.JOB, 'generated', 'baseline') if G.TASK.CLEAN else join(G.TASK.PATHS.JOB, 'generated', 'dirty')

    if G.TASK.CLEAN:
        seqs = kit.model.generate(G.TASK.N_SEQS, G.DHPARAMS.MAX_SEQ_LEN, truncate_after_first_stop_token=True, unpredicted_tokens=unpredicted_tokens)
    else:            
        seqs = kit.model.generate(G.TASK.N_SEQS, G.DHPARAMS.MAX_SEQ_LEN)

    for seq in seqs:
        seq_hash = get_seq_hash(seq)
        removed_tokens = ('*-' + unpredicted_tokens) if G.TASK.CLEAN else ''
        seqs_to_fasta([seq], join(output_folder_path, f"{seq_hash}.fasta"), translate=('', '', removed_tokens))

        if G.TASK.SEQ_HASH_FILE_PATH is not None:
            seq_hashes_to_file([seq_hash], G.TASK.SEQ_HASH_FILE_PATH, append=True)

        if G.TASK.CLEAN and G.TASK.RESULTS.TO_LOCH:
            seqs_to_fasta([seq], get_fasta_file_path(seq_hash))