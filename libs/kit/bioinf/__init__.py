import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import hashlib
import re
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.Polypeptide import three_to_one

import torch
import torch.nn.functional as F


# adapted from Bio.PDB.Polypeptide

# special amino acid values
# -... empty position
# X... unknown
# ?... masked language learning

# https://www.ddbj.nig.ac.jp/ddbj/code-e.html

aa1 = "ACDEFGHIKLMNPQRSTVWYU*-X?"
# aa1 = "ACDEFGHIKLMNPQRSTVWYUO*-X?"
aa1_aliases = {'B': 'N', 'Z': 'Q'}
aa1_std = aa1[:20]

unpredicted_tokens = 'UX?'  # the tokens that our models should not predict

aa3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU",      # standard AAs
       "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR",      # standard AAs
       "SEC", # "PYL"         # additional proteinogenic AAs - https://www.ebi.ac.uk/chebi/searchId.do?printerFriendlyView=true&chebiId=83813&structureView=
       "STP",                 # stop token
       "---", "XXX", "???"]   # special values (None, Unknown, Masked)
n_aa = len(aa1)
n_aa_std = len(aa1_std)

aa_1_to_idx = {}
aa_idx_to_1 = {}
aa_3_to_idx = {}
aa_idx_to_3 = {}
aa_3_to_1 = {}
aa_1_to_3 = {}

# Create some lookup tables
for i in range(0, n_aa):
    aa_1 = aa1[i]
    aa_3 = aa3[i]

    aa_1_to_idx[aa_1] = i
    aa_idx_to_1[i] = aa_1
    aa_3_to_idx[aa_3] = i
    aa_idx_to_3[i] = aa_3
    aa_3_to_1[aa_3] = aa_1
    aa_1_to_3[aa_1] = aa_3


def only_std_aa(seqs, ignore=''):
    if type(seqs) == list:
        result = []
        for seq in seqs:
            result.append(only_std_aa(seq, ignore))
        return result
    else:
        check = aa1_std + ignore
        for letter in seqs:
            if letter not in check:
                return False
        return True


def get_kmers(seqs, lengths, check_aa=True):
    """ returns a set with all kmers of given lengths in the sequnces

    :param seqs: all sequences to consider (list or a single string)
    :param lengths: all kmer lengths to consider (list or single number)
    :param check_aa: if True, only kmers with valid amino acids are returned
    :return: set of kmers in the sequences
    """

    result = set()
    if type(seqs) == list:
        for seq in seqs:
            result = result.union(get_kmers(seq, lengths, check_aa))
    else:
        seq = seqs
        if type(lengths) == list:
            lengths = lengths.copy()
            length = lengths.pop()
            result = get_kmers(seq, length, check_aa)
            if len(lengths) > 0:
                result = result.union(get_kmers(seq, lengths, check_aa))
        else:
            length = lengths
            result = set([str(seq[x: x + length]) for x in range(len(seq) - length + 1)])
            if check_aa:
                result = set([x for x in result if all([y in aa1_std for y in x])])

    return result


def seq_to_kmers_list(seq, k):
    kmers = []
    for pos in range(len(seq) - k + 1):
        kmers.append(seq[pos:pos+k])
    return kmers


def generate_random_peptide(length):
    return generate_random_aa_seq(length)


def generate_random_aa_seq(length):
    return "".join(np.random.choice(list(aa1_std), length))


def pdb_to_seq(pdbfile):
    parser = PDBParser()
    structure = parser.get_structure("structure", pdbfile)
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    residue_name = residue.get_resname()
                    seq += three_to_one(residue_name)
    return seq

def seq_to_idx(seq):
    seq = [(aa if aa not in aa1_aliases else aa1_aliases[aa]) for aa in seq]  # replace aliases
    idx = [aa_1_to_idx[aa] for aa in seq]
    return idx


def seqs_to_tensor(seqs, start="*", end="*", device=None):
    l_seqs, l_masks, tmp = [], [], []

    max_len = 0
    for seq in seqs:
        seq = start + seq + end
        tmp.append(seq)
        max_len = max(max_len, len(seq))

    for seq in tmp:
        l_seqs.append(torch.tensor(seq_to_idx(seq + "-" * (max_len - len(seq)))))
        l_masks.append(torch.tensor([1] * len(seq) + [0] * (max_len - len(seq))))

    result = (torch.stack(l_seqs, dim=0), torch.stack(l_masks, dim=0))
    if device is not None:
        result = (result[0].to(device), result[1].to(device))
    return result


def idx_to_seq(idx):
    return "".join([aa_idx_to_1[i.item()] for i in idx])


def tensor_to_seqs(tensor):
    result = []
    for t in tensor:
        result.append(idx_to_seq(t))
    return result


def get_pwm(peptides, letter=True):
    pwm = {}

    for peptide in peptides:
        if letter:
            peptide = peptide.replace('-', '')
        else:
            peptide = peptide[peptide != 0]

        peptide_len = len(peptide)
        if peptide_len not in pwm:
            pwm[peptide_len] = np.zeros((len(aa1), peptide_len))

        for pos, aa in enumerate(peptide):
            aa_idx = aa_1_to_idx[aa.upper()] if letter else aa
            pwm[peptide_len][aa_idx, pos] += 1

    for key, value in pwm.items():
        pwm[key] = value / value.sum(axis=0)

    return pwm


def seqs_to_df(seqs, accessions):
    dict_seqs = defaultdict(lambda: [0, []])
    for j, seq in enumerate(seqs):
        dict_seqs[seq][0] += 1
        if type(accessions) == str:
            dict_seqs[seq][1] = accessions
        else:
            dict_seqs[seq][1].append(accessions[j])

    df = pd.DataFrame(index=list(dict_seqs.keys()), columns=["cnt", "accession"], data=list(dict_seqs.values()))
    df["length"] = df.apply(lambda row: len(row.name), axis=1)
    return df


def seq_dict_to_df(d_sequences):
    seqs = []
    accessions = []
    cnts = []
    lengths = []
    complete = []
    stdaa = []
    for key, value in d_sequences.items():
        seqs.append(key)
        accessions.append(value)
        cnts.append(len(value))
        lengths.append(len(key))
        complete.append('X' not in key)
        stdaa.append(all([r in (aa1_std + '*') for r in key]))

    return pd.DataFrame({
        "seq": seqs, "length": lengths, "accession": accessions, "cnt": cnts, "complete": complete, "stdaa": stdaa}).set_index("seq")

    
def get_mutations(seq_from, seq_to):
    substitutions, indels = [], []
    for pos, (s_to, s_from) in enumerate(zip(seq_to, seq_from)):
        if s_to != s_from:
            l = substitutions if s_to != '-' and s_from != '-' else indels
            l.append(f"{s_from}{pos + 1}{s_to}")
    return substitutions, indels


def apply_substitutions(seq_from, substitutions, first_pos=1):
    if substitutions == '':
        return seq_from

    seq_to = [a for a in seq_from]
    substitutions = substitutions.split(",")
    for substitution in substitutions:
        aa_from, aa_pos, aa_to = re.findall("^([A-Z*\-]+)(\d+)([A-Z*\-]+)$", substitution)[0]
        aa_idx = int(aa_pos) - first_pos
        assert aa_from == seq_to[aa_idx]
        seq_to[aa_idx] = aa_to
    return ''.join(seq_to)

