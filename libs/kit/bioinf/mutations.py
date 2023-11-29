"""This module contains functions for working with mutations."""

import re


def get_mutations(seq_from, seq_to):
    """Returns a list of mutations between two aligned sequences.

    :param seq_from: str - sequence to mutate from
    :param seq_to: str - sequence to mutate to
    :return: (list, list) - substitutions, indels
    """

    substitutions, indels = [], []
    if len(seq_from) != len(seq_to):
        raise Exception("Sequences must have the same length.")

    for pos, (s_to, s_from) in enumerate(zip(seq_to, seq_from)):
        if s_to != s_from:
            l = substitutions if s_to != "-" and s_from != "-" else indels
            l.append(f"{s_from}{pos + 1}{s_to}")
    return substitutions, indels


def apply_substitutions(seq_from, substitutions, first_pos=1):
    """Applies the given substitutions to the sequence.

    :param seq_from: str - sequence to mutate from
    :param substitutions: str - substitutions to apply (e.g. A1G,C2D)
    :param first_pos: int - first position in the sequence
        (often 1, but can differ dependent on the convention)
    """

    if substitutions == "":
        return seq_from

    seq_to = list(seq_from)
    substitutions = substitutions.split(",")
    for substitution in substitutions:
        aa_from, aa_pos, aa_to = re.findall(
            r"^([A-Z*\-]+)(\d+)([A-Z*\-]+)$", substitution
        )[0]
        aa_idx = int(aa_pos) - first_pos
        assert aa_from == seq_to[aa_idx]
        seq_to[aa_idx] = aa_to
    return "".join(seq_to)
