import re
import os

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from collections import defaultdict
from tqdm.auto import tqdm
import pandas as pd

from kit.loch.utils import get_seq_hash
from kit.bioinf import seq_dict_to_df
from kit.data import str_to_file


def read_folder(folder, encoding=None, stop_token=True, return_df=False, evaluate=False):
    fastanames = []
    for dirname, _, filenames in os.walk(folder, followlinks=True):
        for filename in filenames:
            h = re.findall(f"^(.+)\.fasta$", filename)
            if len(h) == 1:
                fastanames.append(os.path.join(dirname, filename))

    if len(fastanames) > 0:
        return read_fasta(fastanames, encoding, stop_token, return_df, evaluate)
    return None


def read_fasta(filenames, encoding=None, stop_token=True, return_df=False, evaluate=False):
    if type(filenames) == str:
        filenames = [filenames]

    sequences = defaultdict(lambda: [])
    for filename in filenames:
        file = open(filename, "r", encoding=encoding)

        for seq_record in SeqIO.parse(file, "fasta"):
            seq = str(seq_record.seq)
            if stop_token and len(seq) > 0 and seq[-1] != '*':
                seq += "*"
            description = eval(seq_record.description) if evaluate else seq_record.description
            if type(description) == list:
                sequences[seq] += description
            else:
                sequences[seq].append(str(description))

    d_sequences = dict(sequences)

    if return_df:
        return seq_dict_to_df(d_sequences)
    else:
        return d_sequences


def write_fasta(filename, df_sequences, translate=('', '', '')):
    """Write a dataframe with sequences to a fasta file.

    :param filename: str - path to output file
    :param df_sequences: pd.DataFrame - dataframe with sequences
    :param translate: tuple - (src, tgt, None) for translation of sequences"""
    seqs = []
    for seq, row in df_sequences.iterrows():
        new_seq = seq.translate(str.maketrans(*translate))
        new_id = str(row.accession)
        seqs.append(
            SeqRecord(
                Seq(new_seq),
                id=new_id,
                description=""
            )
        )

    SeqIO.write(seqs, filename, format="fasta")


def seqs_to_fasta(seqs, filename, accessions=None, translate=('', '', '')):
    """Write a list of sequences to a fasta file.

    :param filename: str - path to output file
    :param seqs: list - list of sequences
    :param translate: tuple - (src, tgt, None) for translation of sequences"""

    if type(seqs) == str:
        seqs = [seqs]
        if accessions is not None:
            accessions = [accessions]

    if accessions is None:
        accessions = [get_seq_hash(seq) for seq in seqs]

    df = pd.DataFrame({'seq': seqs, 'accession': accessions}).set_index('seq')
    write_fasta(filename, df, translate=translate)


def fastas_to_seqs(fasta_file_paths):
    d_seqs = read_fasta(fasta_file_paths, return_df=False, evaluate=False)
    return list(d_seqs.keys())


def fasta_to_dict(fasta_file_path, encoding=None):
    file = open(fasta_file_path, "r", encoding=encoding)

    sequences = defaultdict(lambda: [])
    for seq_record in SeqIO.parse(file, "fasta"):
        seq = str(seq_record.seq)
        description = seq_record.description
        sequences[seq].append(str(description))

    return dict(sequences)
    
    
def fasta_to_seq_hashes(fasta_file_path, seq_hash_file_path, append=False, output_directory_path=None):
    df = read_fasta(fasta_file_path, return_df=True)
    seq_hashes = []
    for seq, row in df.iterrows():
        seq_hash = get_seq_hash(seq)
        if output_directory_path is not None:
            seqs_to_fasta(seq, os.path.join(output_directory_path, f"{seq_hash}.fasta"), seq_hash)

        seq_hashes.append(seq_hash)

    content = "\n".join(seq_hashes)
    if append:
        content = f"\n{content}"
    str_to_file(content, seq_hash_file_path, append=append)


def translate_fasta(input_file_path, output_file_path=None, translate=('', '', '')):
    seq = fastas_to_seqs(input_file_path)
    seqs_to_fasta(seq, output_file_path if output_file_path is not None else input_file_path, translate=translate)
