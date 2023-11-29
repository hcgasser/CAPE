import pandas as pd

import kit
import kit.globals as G
from kit.path import join
from kit.data import Split
from kit.loch import seq_hashes_to_file
from kit.loch.utils import get_seq_hash
from kit.bioinf.fasta import seqs_to_fasta, read_fasta
from kit.loch.path import get_fasta_file_path


def run():
    if G.TASK.ID == "data_raw_to_input":
        kit.load_dhparams(G.TASK.DHPARAMS, doc_idx=1)
        G.DHPARAMS.CLS.raw_to_input(G.DHPARAMS)

        # copies the clean fasta files to the protein files folder
        for split in ["TRAIN", "VAL", "TEST"]:
            seq_hashes_file_path = join(
                G.PROJECT_ENV.ARTEFACTS, f"{G.DOMAIN}.data.{split}.seq_hash"
            )
            print(split)
            filename = join(G.DHPARAMS.INPUT_FOLDER, f"{split}.fasta")
            df = read_fasta(filename, return_df=True, evaluate=True)

            seq_hashes = []
            for seq, _ in df.iterrows():
                seq_hash = get_seq_hash(seq)
                seqs_to_fasta(
                    seq,
                    get_fasta_file_path(seq_hash),
                    seq_hash,
                    translate=("", "", "*-"),
                )
                seq_hashes.append(seq_hash)

            seq_hashes_to_file(seq_hashes, seq_hashes_file_path)

    if G.TASK.ID == "create_support":
        kit.load_dhparams(G.TASK.DHPARAMS)

        dfs = []
        for split in Split:
            if str(split) in G.DHPARAMS:
                dfs.append(
                    kit.bioinf.fasta.read_fasta(
                        join(G.ENV.INPUT, G.DHPARAMS[str(split)]),
                        evaluate=True,
                        return_df=True,
                    )
                )

        dfs = pd.concat(dfs).query("complete == True")
        df_support = dfs.sample(G.TASK.N_SUPPORTS, replace=False, weights="cnt")

        supports_seq_fasta_file_path = join(
            G.PROJECT_ENV.ARTEFACTS, f"{G.DOMAIN}.support.fasta"
        )
        kit.bioinf.fasta.write_fasta(supports_seq_fasta_file_path, df_support)

        supports_seq_hashes_file_path = join(
            G.PROJECT_ENV.ARTEFACTS, f"{G.DOMAIN}.support.seq_hash"
        )

        seq_hashes = []
        for seq, _ in df_support.iterrows():
            seq_hash = get_seq_hash(seq)
            seq_hashes.append(seq_hash)

        seq_hashes_to_file(seq_hashes, supports_seq_hashes_file_path)
