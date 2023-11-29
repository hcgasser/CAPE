""" Functions for handling kmers """

import numpy as np
from tqdm.auto import tqdm


def get_visible_mhc_1_peptides(
    df_eval, predictor_mhc_1, mhc_1_alleles, mhc_1_peptide_lengths=None
):
    """Get all visible peptides for each sequence in df_eval"""
    d_visible = {}
    for _, row in tqdm(df_eval.iterrows()):
        v = []
        visible = predictor_mhc_1.seq_presented(
            row.seq, mhc_1_alleles, mhc_1_peptide_lengths
        )
        for kmer, _, _, _ in visible:
            v.append(kmer)

        d_visible[row.seq_hash] = set(v)

    return d_visible


def get_metrics(d_visible, seq_hash):
    """Get precision and recall values for a given sequence

    The d_visible dictionary specifies the sequences to compare to as well as
    the visible peptides for each sequence. The seq_hash specifies the query sequence
    to compare to all other sequences in d_visible.

    :param d_visible: dictionary of sequences and their visible peptides
    :param seq_hash: sequence hash of the query sequence
    :return: dictionaries of precision and recall values
    """

    d_precision = {}
    d_recall = {}
    visible = d_visible[seq_hash]
    for seq_hash_, visible_ in d_visible.items():
        l_intersection = len(visible.intersection(visible_))
        d_precision[seq_hash_] = l_intersection / len(visible)
        d_recall[seq_hash_] = l_intersection / len(visible_)

    return d_precision, d_recall


def add_metrics(seq_hash, df_eval, d_visible, peptide_lengths_col):
    d_precision, d_recall = get_metrics(d_visible, seq_hash)
    df_eval[f"precision_{peptide_lengths_col}_{seq_hash}"] = df_eval.seq_hash.map(
        d_precision
    )
    """ add recall and precision to df_eval
    
    :param seq_hash: sequence hash of the query sequence
    :param df_eval: dataframe of sequences
    :param d_visible: dictionary of sequences and their visible peptides
    :param peptide_lengths_col: column name of the peptide lengths
    :return: None
    """

    df_eval[f"recall_{peptide_lengths_col}_{seq_hash}"] = df_eval.seq_hash.map(d_recall)


# def add_recall(df_eval, df_seq_kmers, predictor_MHC_I, MHCs, override=False):
#     col = f'recall_{df_seq_kmers.seq_hash}'
#     if col not in df_eval.columns or override or df_eval[col].count() != len(df_eval):
#         df_eval[col] = None
#         for idx, row in df_eval.iterrows():
#             visible = predictor_MHC_I.seq_presented(row.seq, MHCs, lengths=[9])
#             recalled = 0
#             for kmer, _, _, _ in visible:
#                 if kmer in df_seq_kmers.index:
#                     recalled += 1
#             df_eval.at[idx, col] = recalled / len(visible)


# def add_precision(df_eval, df_seq_kmers, details=False):
#     col = f'precision_{df_seq_kmers.seq_hash}' + ('_d' if details else '')
#     epitopes = list(df_seq_kmers.query('presented').index)
#     df_eval[col] = None
#     for idx, row in df_eval.iterrows():
#         found = []
#         for epitope in epitopes:
#             if epitope in row.seq:
#                 found.append(epitope)

#         result = found if details else len(found)/len(epitopes)
#         df_eval.at[idx, col] = result


def add_precision_to_seq_kmers(df_seq_kmers, natural_kmers_data, consider_one=True):
    natural_kmers_dict, n_natural = (
        natural_kmers_data["natural_kmers_dict"],
        natural_kmers_data["n_natural"],
    )

    df_seq_kmers["precision"] = None
    for kmer, _ in df_seq_kmers.iterrows():
        n_kmer_natural = natural_kmers_dict[len(kmer)].get(kmer, 0)
        precision = n_kmer_natural / n_natural
        if n_kmer_natural == 1 and not consider_one:
            precision = 0
        df_seq_kmers.at[kmer, "precision"] = precision


# potential test procedure
# def add_precision_to_seq_kmers_alt(df_seq_kmers, df_eval):
#     df_seq_kmers['precision_alt'] = None
#     add_precision(df_eval, df_seq_kmers, details=True)

#     epitopes = list(df_seq_kmers.query('presented').index)
#     epitopes = {epi: 0 for epi in epitopes}

#     df_natural = df_eval.query('source == "natural"')
#     for epis in df_natural[f'precision_{df_seq_kmers.seq_hash}_d']:
#         for epi in epis:
#             epitopes[epi] += 1

#     epitopes = {key: value/len(df_natural) for key, value in epitopes.items()}

#     for epi, val in epitopes.items():
#         df_seq_kmers.at[epi, 'precision_alt'] = val


def calc_recall_metrics(df_eval, seq_hashes, lengths):
    if isinstance(seq_hashes, list):
        seq_hashes = [seq_hashes]

    col = "+".join([str(x) for x in lengths])

    no_recalls, mean_recalls = [], []
    for seq_hash in seq_hashes:
        no_recalls.append(
            len(
                df_eval.query(f'source == "natural" and recall_{col}_{seq_hash} == 0.0')
            )
            / len(df_eval.query('source == "natural"'))
        )
        mean_recalls.append(
            df_eval.query('source == "natural"')[f"recall_{col}_{seq_hash}"].mean()
        )

    mean_recall = np.mean(mean_recalls)
    no_recall = np.mean(no_recalls)

    return {
        "mean_recall": mean_recall,
        "no_recall": no_recall,
    }
