import os

import numpy as np

import kit.globals as G
from kit.bioinf import get_kmers
from kit.log import log_info
from kit.path import join
from kit.data import Split, str_to_file, file_to_str
from kit.bioinf.fasta import fastas_to_seqs


d_visible_natural = {}


def get_visible_natural(predictor, alleles, kmer_lengths, input_files_path):
    global d_visible_natural

    key = tuple(alleles + kmer_lengths)
    if key in d_visible_natural:
        return d_visible_natural[key]

    visible_file_path = join(
        input_files_path,
        "visible",
        f'{"+".join(alleles)}_{"+".join([str(l) for l in kmer_lengths])}.txt',
    )
    if not os.path.exists(visible_file_path):
        natural_seqs = []
        visible_natural = set()
        for split in Split:
            split_file_path = join(input_files_path, f"{split}.fasta")
            if os.path.exists(split_file_path):
                natural_seqs += fastas_to_seqs(split_file_path)

        for natural_seq in natural_seqs:
            visible_natural = visible_natural.union(
                [
                    v[0]
                    for v in predictor.seq_presented(
                        natural_seq, alleles, lengths=kmer_lengths
                    )
                ]
            )

        str_to_file("\n".join(list(visible_natural)), visible_file_path)
    else:
        visible_natural = set(file_to_str(visible_file_path).split("\n"))
        log_info(f"loaded {len(visible_natural)} visible peptides from file")

    d_visible_natural[key] = visible_natural

    return visible_natural


def rewards_seqs(
    predictor,
    seqs,
    alleles,
    profile,
    mhc_1_peptide_lengths,
    rwd_impossible_token=-1,
    rwd_premature_stop=None,
    rwd_missing_stop=None,
    return_details=False,
):
    """returns a matrix of rewards for each position in each sequence

    The reward for an impossible token is given once.
        Afterwards it is removed from the sequence
    The reward for a premature stop token is given once
        after the next token (which is not '-') got added.
        Afterwards it is removed from the sequence

    :param predictor: predictor object with a predict method
    :param seqs: sequences to evaluate
    :param alleles: alleles to evaluate
    :param profile: CAPE.profile.vis_down or CAPE.profile.vis_up
    :param MHC_1_PEPTIDE_LENGTHS: lengths of peptides to consider
    :param rwd_impossible_token: reward for a peptide with an impossible token
    :param rwd_premature_stop: reward for a peptide that ends prematurely
    :param rwd_missing_stop: reward for a peptide that is missing a stop token
    :param return_details: return the peptides that were checked (for testing)

    :return: matrix of rewards for each position in each sequence"""

    # get the rewards
    reward_visible_artificial = profile.reward_visible_artificial()
    reward_visible_natural = profile.reward_visible_natural()

    if reward_visible_natural != reward_visible_artificial:
        visible_natural = get_visible_natural(
            predictor,
            alleles,
            mhc_1_peptide_lengths,
            os.path.join(G.ENV.INPUT, G.DHPARAMS.FOLDER),
        )
    else:
        visible_natural = set()

    # remove impossible tokens to find all peptides that need to be assessed
    translation = str.maketrans(*("", "", "U*-X?"))
    clean_seqs = [seq.translate(translation) for seq in seqs]

    kmers = get_kmers(clean_seqs, mhc_1_peptide_lengths)
    # check which kmers are currently unknown
    for kmer in kmers:
        for allele in alleles:
            if kmer not in predictor.percentiles[allele]:
                if not any(l in kmer for l in "U*-X?"):
                    predictor.missing[allele].append(kmer)

    # get the missing predictions
    predictor.predict_missing_peptides()

    max_len = np.max([len(seq) for seq in seqs])

    min_reward = -(len(alleles) * len(mhc_1_peptide_lengths))

    rwd_impossible_token = (
        min_reward if rwd_impossible_token is None else rwd_impossible_token
    )
    rwd_premature_stop = (
        min_reward if rwd_premature_stop is None else rwd_premature_stop
    )
    rwd_missing_stop = min_reward if rwd_missing_stop is None else rwd_missing_stop

    rewards = np.zeros((len(seqs), max_len))
    rewards_natural = np.zeros((len(seqs), max_len))
    rewards_artificial = np.zeros((len(seqs), max_len))

    checked_peptides = []
    for s, seq in enumerate(seqs):
        seq_len = len(seq)

        stopped = False
        j, t = 0, 0
        while t < seq_len:
            while stopped and t < seq_len:
                if seq[j] != "-":
                    stopped = False
                    rewards[s, t] += rwd_premature_stop
                else:
                    seq = seq[:j] + seq[j + 1 :]
                    t += 1

            if t < seq_len:
                if seq[j] == "*":
                    seq = seq[:j] + seq[j + 1 :]
                    stopped = True
                else:
                    if seq[j] in "U*-X?":
                        rewards[s, t] += rwd_impossible_token
                        seq = seq[:j] + seq[j + 1 :]
                    else:
                        for length in [l for l in mhc_1_peptide_lengths if l <= j]:
                            peptide = seq[j + 1 - length : j + 1]
                            checked_peptides.append(peptide)
                            for allele in alleles:
                                if predictor.peptide_presented(peptide, allele):
                                    if peptide in visible_natural:
                                        rewards_natural[s, t] += reward_visible_natural
                                    else:
                                        rewards_artificial[
                                            s, t
                                        ] += reward_visible_artificial
                        j += 1

                t += 1

        # sequences should still end in stop tokens
        if not stopped:
            rewards[s, -1] += rwd_missing_stop

    rewards = rewards + rewards_natural + rewards_artificial
    if return_details:
        return rewards, rewards_natural, rewards_artificial, checked_peptides
    return rewards
