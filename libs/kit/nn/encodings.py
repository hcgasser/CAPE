import pandas as pd
import os
import math
import torch
from torch import nn, Tensor

from kit.log import log_info, log_caller
from kit.data import scale_df
from kit.bioinf import aa1_std, aa1


def create_embedding(weights, trainable=False):
    """ creates a torch.nn.Embedding object using the specified weights

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


def load_aa_encodings(folder):
    log_caller()

    result = {}

    filename = os.path.join(folder, "AA8.csv")
    if os.path.exists(filename):
        df_AA8 = pd.read_csv(filename).set_index("AA")
        std_aa1 = [aa for aa in aa1_std]
        mean = df_AA8.loc[std_aa1, std_aa1].to_numpy().mean()
        std = df_AA8.loc[std_aa1, std_aa1].to_numpy().std()
        df_AA8.loc[std_aa1, std_aa1] = (df_AA8.loc[std_aa1, std_aa1] - mean) / std
        result['AA8'] = df_AA8

    filename = os.path.join(folder, "PMBEC.csv")
    if os.path.exists(filename):
        result['PMBEC'] = pd.read_csv(os.path.join(folder, "PMBEC.csv")).set_index("AA")

    filename = os.path.join(folder, "VHSE.csv")
    if os.path.exists(filename):
        result['VHSE'] = pd.read_csv(os.path.join(folder, "VHSE.csv")).set_index("AA")

    filename = os.path.join(folder, "BLOSUM62.csv")
    if os.path.exists(filename):
        result['BLOSUM62'] = scale_df(pd.read_csv(os.path.join(folder, "BLOSUM62.csv")).set_index("AA"))

    return result


class AAEmbedding(torch.nn.Module):
    def __init__(self, encodings, folder):
        super(self.__class__, self).__init__()
        log_caller(self)

        self.encoding_list = torch.nn.ModuleList([])
        self.total_dim = 0

        dict_aa_encodings = load_aa_encodings(folder)

        for encoding in encodings.split("+"):
            if encoding != '':
                if encoding.isnumeric():
                    dim = int(encoding)
                    one_hot = torch.nn.Embedding(len(aa1), dim)
                    self.encoding_list.append(one_hot)
                    log_info(f"\tAdded trained AA embedding with {dim} dims")
                else:
                    weights = torch.tensor(dict_aa_encodings[encoding].to_numpy(), dtype=torch.float32)
                    self.encoding_list.append(create_embedding(weights, False))
                    dim = weights.shape[1]
                    log_info(f"\tAdded fixed AA {encoding} encoding with {dim} dims")
                self.total_dim += dim

    def forward(self, x):
        encoding = []
        for embedding in self.encoding_list:
            encoding.append(embedding(x))

        return torch.cat(encoding, axis=-1)


class PositionalEncoding(nn.Module):
    """
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
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, pos) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim_aa + embedding_dim_type]
            pos: Tensor, shape [batch_size, seq_len]
            type: InputType

        Return:
            y: Tensor, shape [bath_size, seq_len, embedding_dim_aa + embedding_dim_type + embedding_dim_pos]
        """

        pe = self.pe[:, :x.size(-2), :].to(device=x.device)
        pe = pe.broadcast_to((x.size(0), pe.size(1), pe.size(2)))
        return torch.cat([x, pe], dim=-1)


class LearnedTokenEmbedding(nn.Module):
    def __init__(self, block_size, n_tokens, d_model, n_positions_embedding=True):
        super().__init__()
        self.d_model = d_model
        self.token_embedding_table = nn.Embedding(n_tokens, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.n_position_embedding_table = nn.Embedding(block_size, d_model) if n_positions_embedding else None

    def forward(self, x=None, positions=None, n_positions=None):
        B, T = x.shape if x is not None else n_positions.shape
        device = x.device if x is not None else n_positions.device

        result = torch.zeros((B, T, self.d_model), device=device)  # (B,T,C)
        if x is not None:
            result += self.token_embedding_table(x)  # add token embedding
            if positions is None:
                positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            result += self.position_embedding_table(positions)

        if n_positions is not None:
            result += self.n_position_embedding_table(n_positions)

        return result
