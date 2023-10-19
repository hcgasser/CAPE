import numpy as np

from kit.bioinf import get_kmers


def rewards_seqs(predictor, seqs, alleles, direction, 
                 lengths=[8, 9, 10], 
                 rwd_impossible_token=-1,
                 rwd_premature_stop=None, 
                 rwd_missing_stop=None,
                 return_checked_peptides=False):
    """ returns a matrix of rewards for each position in each sequence

    The reward for an impossible token is given once. 
        Afterwards it is removed from the sequence
    The reward for a premature stop token is given once after the next token (which is not '-') got added. 
        Afterwards it is removed from the sequence

    :param predictor: predictor object with a predict method
    :param seqs: sequences to evaluate
    :param alleles: alleles to evaluate
    :param direction: 'deimmunize' or 'immunize'
    :param lengths: lengths of peptides to consider
    :param rwd_impossible_token: reward for a peptide with an impossible token
    :param rwd_premature_stop: reward for a peptide that ends prematurely
    :param rwd_missing_stop: reward for a peptide that is missing a stop token
    :param return_checked_peptides: return the peptides that were checked (for testing)
    
    :return: matrix of rewards for each position in each sequence"""

    impossible_tokens = 'U*-X?'

    # remove impossible tokens to find all peptides that need to be assessed
    translation = str.maketrans(*('', '', 'U*-X?'))
    clean_seqs = [seq.translate(translation) for seq in seqs]

    kmers = get_kmers(clean_seqs, lengths)
    # check which kmers are currently unknown
    for kmer in kmers:
        for allele in alleles:
            if kmer not in predictor.percentiles[allele]:
                if not any([l in kmer for l in 'U*-X?']):
                    predictor.missing[allele].append(kmer)

    # get the missing predictions
    predictor.predict_missing_peptides()

    max_len = np.max([len(seq) for seq in seqs])

    # get the rewards
    direction = -1. if direction =="deimmunize" else 1.
    min_reward = -(len(alleles) * len(lengths))

    rwd_impossible_token = min_reward if rwd_impossible_token is None else rwd_impossible_token  
    rwd_premature_stop = min_reward if rwd_premature_stop is None else rwd_premature_stop
    rwd_missing_stop = min_reward if rwd_missing_stop is None else rwd_missing_stop

    rewards = np.zeros((len(seqs), max_len))
    max_peptide_length = np.max(lengths)

    checked_peptides = []
    for s, seq in enumerate(seqs):
        seq_len = len(seq)

        stopped = False
        j, t = 0, 0
        while t < seq_len:
            while stopped and t < seq_len:
                if seq[j] != '-':
                    stopped = False
                    rewards[s, t] += rwd_premature_stop
                else:
                    seq = seq[:j] + seq[j+1:]
                    t += 1

            if t < seq_len:
                if seq[j] == '*':
                    seq = seq[:j] + seq[j+1:]
                    stopped = True
                else:
                    if seq[j] in 'U*-X?':
                        rewards[s, t] += rwd_impossible_token
                        seq = seq[:j] + seq[j+1:]
                    else:
                        for length in [l for l in lengths if l <= j]:
                            peptide = seq[j+1-length:j+1]
                            checked_peptides.append(peptide)
                            for allele in alleles:
                                if predictor.peptide_presented(peptide, allele):
                                    rewards[s, t] += direction
                        j += 1

                t += 1


        # sequences should still end in stop tokens
        if not stopped:
            rewards[s, -1] += rwd_missing_stop

    if return_checked_peptides:
        return rewards, checked_peptides
    return rewards

import kit
