"""This module contains functions for working with encodings/embeddings."""

import os
import math

import pandas as pd

import torch
from torch import nn, Tensor

from kit.log import log_info, log_caller
from kit.data.utils import scale_df
from kit.bioinf import AA1_STD, AA1_FULL


def create_embedding(weights, trainable=False):
    """creates a torch.nn.Embedding object using the specified weights

    source: https://discuss.pytorch.org/t/set-weights-for-embedding-layer/56097

    :param weights: the weights to be used in the torch.nn.Embedding layer
    :param trainable: should the weights in the returned layer be trainable or not?
    :return: the constructed torch.nn.Embedding layer
    """

    num_embeddings, embedding_dim = weights.shape
    embedding = torch.nn.Embedding(num_embeddings, embedding_dim)
    embedding.weight = torch.nn.Parameter(weights)
    if trainable:
        embedding.weight.requires_grad = True
    else:
        embedding.weight.requires_grad = False
    return embedding


def load_aa_encodings(dir_path):
    """Loads AA encodings from the specified directory path.

    The following files are looked for in the directory:

    1. AA8.csv
    2. PMBEC.csv
    3. VHSE.csv
    4. BLOSUM62.csv

    :param dir_path: str - path to the directory containing the encodings
    :return: dict - AA encodings
    """

    log_caller()

    result = {}

    filename = os.path.join(dir_path, "AA8.csv")
    if os.path.exists(filename):
        df_aa8 = pd.read_csv(filename).set_index("AA")
        l_aa1_std = list(AA1_STD)
        mean = df_aa8.loc[l_aa1_std, l_aa1_std].to_numpy().mean()
        std = df_aa8.loc[l_aa1_std, l_aa1_std].to_numpy().std()
        df_aa8.loc[l_aa1_std, l_aa1_std] = (
            df_aa8.loc[l_aa1_std, l_aa1_std] - mean
        ) / std
        result["AA8"] = df_aa8

    filename = os.path.join(dir_path, "PMBEC.csv")
    if os.path.exists(filename):
        result["PMBEC"] = pd.read_csv(os.path.join(dir_path, "PMBEC.csv")).set_index(
            "AA"
        )

    filename = os.path.join(dir_path, "VHSE.csv")
    if os.path.exists(filename):
        result["VHSE"] = pd.read_csv(os.path.join(dir_path, "VHSE.csv")).set_index("AA")

    filename = os.path.join(dir_path, "BLOSUM62.csv")
    if os.path.exists(filename):
        result["BLOSUM62"] = scale_df(
            pd.read_csv(os.path.join(dir_path, "BLOSUM62.csv")).set_index("AA")
        )

    return result


class AAEmbedding(torch.nn.Module):
    """Embedding of amino acids.

    :param encodings: str - encodings to use, e.g. "32+AA8+PMBEC+VHSE+BLOSUM62"
        32... one-hot encoding with 32 dimensions
        AA8... AA8 encoding
        PMBEC... PMBEC encoding
        VHSE... VHSE encoding
        BLOSUM62... BLOSUM62 encoding
    :param dir_path: str - path to the directory containing the encodings
    """

    def __init__(self, encodings, dir_path):
        super().__init__()
        log_caller(self)

        self.encoding_list = torch.nn.ModuleList([])
        self.total_dim = 0

        dict_aa_encodings = load_aa_encodings(dir_path)

        for encoding in encodings.split("+"):
            if encoding != "":
                if encoding.isnumeric():
                    dim = int(encoding)
                    one_hot = torch.nn.Embedding(len(AA1_FULL), dim)
                    self.encoding_list.append(one_hot)
                    log_info(f"\tAdded trained AA embedding with {dim} dims")
                else:
                    weights = torch.tensor(
                        dict_aa_encodings[encoding].to_numpy(), dtype=torch.float32
                    )
                    self.encoding_list.append(create_embedding(weights, False))
                    dim = weights.shape[1]
                    log_info(f"\tAdded fixed AA {encoding} encoding with {dim} dims")
                self.total_dim += dim

    def forward(self, x):
        """Takes in a tensor of shape [batch_size, seq_len] and returns a tensor of shape
        [batch_size, seq_len, embedding_dim_aa]"""

        encoding = []
        for embedding in self.encoding_list:
            encoding.append(embedding(x))

        return torch.cat(encoding, axis=-1)


class PositionalEncoding(nn.Module):
    """Calculated positional encodings

    Adapted from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_pos: int, max_len: int = 5000):
        super().__init__()
        self.d_pos = d_pos

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_pos, 2) * (-math.log(10000.0) / d_pos))
        pe = torch.zeros(1, max_len, d_pos)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim_aa + embedding_dim_type]
            type: InputType

        Return:
            y: Tensor, shape [bath_size,
                              seq_len,
                              embedding_dim_aa + embedding_dim_type + embedding_dim_pos]
        """

        pe = self.pe[:, : x.size(-2), :].to(device=x.device)
        pe = pe.broadcast_to((x.size(0), pe.size(1), pe.size(2)))
        return torch.cat([x, pe], dim=-1)


class LearnedTokenEmbedding(nn.Module):
    """Learned token embeddings

    :param block_size: int - block size
    :param n_tokens: int - number of tokens
    :param d_model: int - embedding dimension
    :param next_position_embeddings: bool - should next positions be embedded?
    """

    def __init__(self, max_length, n_tokens, d_model, next_position_embeddings=True):
        super().__init__()
        self.d_model = d_model
        self.token_embedding_table = nn.Embedding(n_tokens, d_model)
        self.position_embedding_table = nn.Embedding(max_length, d_model)
        self.n_position_embedding_table = (
            nn.Embedding(max_length, d_model) if next_position_embeddings else None
        )

    def forward(self, x=None, positions=None, next_positions=None):
        B, T = x.shape if x is not None else next_positions.shape
        device = x.device if x is not None else next_positions.device

        result = torch.zeros((B, T, self.d_model), device=device)  # (B,T,C)
        if x is not None:
            result += self.token_embedding_table(x)  # add token embedding
            if positions is None:
                positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            result += self.position_embedding_table(positions)

        if next_positions is not None:
            result += self.n_position_embedding_table(next_positions)

        return result
