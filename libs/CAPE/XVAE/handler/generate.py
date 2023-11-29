""" Handler for the generate task. """

import kit
import kit.globals as G
from kit.loch import seq_hashes_to_file
from kit.loch.utils import get_seq_hash
from kit.loch.path import get_fasta_file_path
from kit.bioinf import unpredicted_tokens
from kit.bioinf.fasta import seqs_to_fasta
from kit.path import join

from CAPE.XVAE import load_ckpt, add_shortcut_to_artefacts


def run():
    """main entrance into the handler"""

    ckpt_id = G.TASK.CKPT_ID
    ckpt_dir_path = load_ckpt(ckpt_id)
    if G.TASK.OUTPUT_FOLDER_PATH is not None:
        output_folder_path = G.TASK.OUTPUT_FOLDER_PATH
    else:
        output_folder_path = (
            join(ckpt_dir_path, "generated", "baseline")
            if G.TASK.CLEAN
            else join(ckpt_dir_path, "generated", "dirty")
        )

    if G.TASK.CLEAN:
        seqs = kit.MODEL.generate(
            G.TASK.N_SEQS,
            G.DHPARAMS.MAX_SEQ_LEN,
            truncate_after_first_stop_token=True,
            unpredicted_tokens=unpredicted_tokens,
        )
    else:
        seqs = kit.MODEL.generate(G.TASK.N_SEQS, G.DHPARAMS.MAX_SEQ_LEN)

    for seq in seqs:
        seq_hash = get_seq_hash(seq)
        removed_tokens = ("*-" + unpredicted_tokens) if G.TASK.CLEAN else ""
        seqs_to_fasta(
            [seq],
            join(output_folder_path, f"{seq_hash}.fasta"),
            translate=("", "", removed_tokens),
        )

        if G.TASK.SEQ_HASH_FILE_PATH != "":
            kind = "final" if G.TASK.CLEAN else "dirty"
            seq_hash_file_path = (
                join(G.ENV.ARTEFACTS, f"{G.DOMAIN}.CAPE-XVAE.baseline.{kind}.seq_hash")
                if G.TASK.SEQ_HASH_FILE_PATH is None
                else G.TASK.SEQ_HASH_FILE_PATH
            )
            seq_hashes_to_file([seq_hash], seq_hash_file_path, append=True)

        if G.TASK.CLEAN and G.TASK.RESULTS.TO_LOCH:
            seqs_to_fasta([seq], get_fasta_file_path(seq_hash))

    add_shortcut_to_artefacts("generated", ckpt_dir_path, ckpt_id)
