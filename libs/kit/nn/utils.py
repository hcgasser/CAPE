import random
from kit.bioinf import aa1_std


def get_masked_sequence(seq, choose=0.15, mask=0.8, wrong=0.1, tokens=None, mask_token='?'):
    """ returns a sequence and a mask in which
    some chosen tokens are masked by a mask, replaced by another token or left unchanged.
    This function is intended to be used in MLM

    :param seq: list of elements to be masked
    :param choose: % of elements to be chosen
    :param mask: % of the chosen elements to be replaced by a mask token
    :param wrong: % ot the chosen elements to be replaced by a wrong token
    :param tokens: a list of tokens constituting the whole alphabet
    :param mask_token: token used to mask elements
    :return: modified sequence and a sequence of 0 (not chosen) and 1 (chosen) signifying the chosen elements
    """

    input_seq = seq.copy()
    seq_len = len(input_seq)
    tokens = tokens if tokens is not None else [x for x in aa1_std]

    # mask tokens
    n_chosen = int(seq_len * choose)
    chosen = random.sample(range(seq_len), k=n_chosen)
    chosen.sort()
    for c in chosen:
        r = random.random()
        if r <= mask:  # mask input token with a probability of 80%
            input_seq[c] = mask_token
        elif r <= (mask + wrong):  # replace input token with a wrong token with a probability of 10%
            input_seq[c] = random.choice([t for t in tokens if t != input_seq[c]])

    return input_seq, [(1 if c in chosen else 0) for c in range(seq_len)], seq


def get_holed_sequence(seqs, hole_tokens, hole_end_tokens, holes=1, wrong=0.1, correct=0.1):

    seq_len = len(seqs[0])

    # sample random hole beginnings
    holes_start = random.sample(range(seq_len+1), k=holes)
    holes_start.sort()

    # sample random hole ends
    holes_end = []
    for j, start in enumerate(holes_start):
        max_end = holes_start[j+1] if j < (len(holes_start)-1) else seq_len + 1
        holes_end.append(random.sample(range(start, max_end), k=1)[0])

    input_seqs, target_seqs = [], []
    for seq, hole_token, hole_end_token in zip(seqs, hole_tokens, hole_end_tokens):  # for each sequence to produce a hole for
        von = 0
        input_seq, chosen, target_seq = [], [], []
        for start, end in zip(holes_start, holes_end):
            input_seq += (seq[von:start].copy() + [hole_token])
            target = seq[start] if start < end else hole_end_token
            target_seq += (seq[von:start].copy() + [target])
            chosen += ([0] * (start - von) + [1])
            von = end

        input_seq += seq[von:].copy()
        target_seq += seq[von:].copy()
        chosen += ([0] * (seq_len - von))

        input_seqs.append(input_seq)
        target_seqs.append(target_seq)

    wrong *= holes
    correct *= holes

    return input_seqs, chosen, target_seqs
