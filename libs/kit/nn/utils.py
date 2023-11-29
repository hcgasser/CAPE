"""This module contains utility functions for neural networks"""

import random
from kit.bioinf import AA1_STD


def get_masked_sequence(
    seq, choose=0.15, mask=0.8, wrong=0.1, tokens=None, mask_token="?"
):
    """returns a sequence and a mask in which
    some chosen tokens are masked by a mask, replaced by another token or left unchanged.
    This function is intended to be used in MLM

    :param seq: list of elements to be masked
    :param choose: % of elements to be chosen
    :param mask: % of the chosen elements to be replaced by a mask token
    :param wrong: % ot the chosen elements to be replaced by a wrong token
    :param tokens: a list of tokens constituting the whole alphabet
    :param mask_token: token used to mask elements
    :return: modified sequence
        and a sequence of 0 (not chosen) and 1 (chosen) signifying the chosen elements
    """

    input_seq = seq.copy()
    seq_len = len(input_seq)
    tokens = tokens if tokens is not None else list(AA1_STD)

    # mask tokens
    n_chosen = int(seq_len * choose)
    chosen = random.sample(range(seq_len), k=n_chosen)
    chosen.sort()
    for c in chosen:
        r = random.random()
        if r <= mask:  # mask input token with a probability of 80%
            input_seq[c] = mask_token
        elif r <= (
            mask + wrong
        ):  # replace input token with a wrong token with a probability of 10%
            input_seq[c] = random.choice([t for t in tokens if t != input_seq[c]])

    return input_seq, [(1 if c in chosen else 0) for c in range(seq_len)], seq
