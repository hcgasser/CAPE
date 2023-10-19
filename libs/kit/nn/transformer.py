"""
    This module is heavily inspired by Andrej Karpathy youtube video:
        "Let's build GPT: from scratch, in code, spelled out. "
        https://www.youtube.com/watch?v=kCc8FmEb1nY

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from kit.nn.encodings import LearnedTokenEmbedding


class LayerNorm1d:
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return (self.gamma, self.beta)


class Head(nn.Module):
    def __init__(self, d_model, head_size, block_size, dropout, masked):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.masked = masked
        if self.masked:
            self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

        self.recalc_masked = False  # when set to True, allows to let gradients flow back to the kv sequence (e.g. latent variable)

    def forward(self, x_q, x_kv, hidden=None):
        B, T, C = x_q.shape

        # produce the query, key and value_net tensors
        q = self.query(x_q)  # (B, T, C)

        # START: hidden state update (and key, value_net calculations - who go into the hidden state)
        # We need to distinguish between
        # - the auto-regressive (masked) case - for which we will always
        #   have to calculate a new k, v pair (and append to the hidden state).
        # - the unrestricted attention case, for which only once the k, v pairs need to be calculated
        if self.masked:
            k = self.key(x_kv)  # (B, _, C)
            v = self.value(x_kv)  # (B, _, C)
            if hidden is not None:  # we need to append the current values to the hidden state
                k = torch.concat([hidden[0], k], dim=-2)  # (B, T, C)
                v = torch.concat([hidden[1], v], dim=-2)  # (B, T, C)
        else:
            if hidden is None or self.recalc_masked:
                k = self.key(x_kv)  # (B, T, C)
                v = self.value(x_kv)  # (B, T, C)
            else:
                k, v = hidden
        # END: update hidden state (and key, value_net calculations)

        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        if self.masked:
            wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out, (k.detach(), v.detach())


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, head_size, block_size, dropout, masked):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, head_size, block_size, dropout, masked) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv=None, hidden=None):
        x_kv = x_kv if x_kv is not None else x_q
        hidden = [None]*len(self.heads) if hidden is None else hidden

        out = [h(x_q, x_kv, hidden[j]) for j, h in enumerate(self.heads)]
        hidden = [o[1] for o in out]
        out = torch.cat([o[0] for o in out], dim=-1)
        out = self.dropout(self.proj(out))
        return out, hidden


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ffd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffd),
            nn.LeakyReLU(),  # was ReLU first
            nn.Linear(d_ffd, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BlockSelfAttention(nn.Module):
    """ Similar to the encoder block in "Attention is all you need"
    however it also supports masking previous sequence elements. If this is done
    the behaviour is like in GPT-2 """

    def __init__(self, name, d_model, d_ffd, n_heads, block_size, dropout, masked):
        super().__init__()
        head_size = d_model // n_heads

        self.name = name
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, n_heads, head_size, block_size, dropout, masked)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ffd, dropout)

    def forward(self, x, hidden=None):
        out_sa, hidden = self.sa(self.ln1(x), hidden)  # order of ln different from original paper
        x = x + out_sa
        x = x + self.ffwd(self.ln2(x))  # order of ln different from original paper
        return x, hidden


class BlockCrossAttention(nn.Module):
    """ Similar to the decoder block in "Attention is all you need" """

    def __init__(self, name, d_model, d_ffd, n_heads, block_size, dropout, masked_sa, masked_xa):
        super().__init__()
        head_size = d_model // n_heads

        self.name = name
        self.ln1 = nn.LayerNorm(d_model)
        self.sa = MultiHeadAttention(d_model, n_heads, head_size, block_size, dropout, masked_sa)
        self.ln2 = nn.LayerNorm(d_model)
        self.xa = MultiHeadAttention(d_model, n_heads, head_size, block_size, dropout, masked_xa)
        self.ln3 = nn.LayerNorm(d_model)
        self.ffwd = FeedForward(d_model, d_ffd, dropout)

    def forward(self, x, x_kv, hidden=None):
        hidden_sa, hidden_xa = (None, None) if hidden is None else hidden
        x_kv = x if x_kv is None else x_kv

        out_sa, hidden_sa = self.sa(self.ln1(x), hidden=hidden_sa)  # order of ln different from original paper
        x = x + out_sa
        out_xa, hidden_xa = self.xa(self.ln2(x), x_kv=x_kv, hidden=hidden_xa)  # order of ln different from original paper
        x = x + out_xa
        x = x + self.ffwd(self.ln3(x))  # order of ln different from original paper
        return x, (hidden_sa, hidden_xa)


class EncoderStack(nn.Module):
    def __init__(self, d_model, d_ffd, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size

        self.blocks = nn.ModuleList(
            [BlockSelfAttention(f"block_{j}", d_model, d_ffd, n_heads, block_size, dropout, masked=False)
             for j in range(n_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x, _ = block(x)
        return x


class DecoderStack(nn.Module):
    def __init__(self, d_model, d_ffd, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.lnX = nn.LayerNorm(d_model)

        self.blocks = nn.ModuleList(
            [BlockCrossAttention(f"block_{j}", d_model, d_ffd, n_heads, block_size, dropout, masked_sa=True, masked_xa=False)
             for j in range(n_layers)]
        )

    def forward(self, x, x_kv, hidden=None):
        hidden = [None]*len(self.blocks) if hidden is None else hidden

        if x_kv is not None:
            x_kv = self.lnX(x_kv)

        for j, block in enumerate(self.blocks):
            x, hidden[j] = block(x, x_kv, hidden[j])

        return x, hidden


class TransformerLM(nn.Module):
    def __init__(self, d_model, d_ffd, n_heads, n_layers, block_size, n_tokens, dropout, input_tokens=True):
        super().__init__()
        self.input_tokens = input_tokens
        self.block_size = block_size

        self.embedding = LearnedTokenEmbedding(block_size, n_tokens, d_model) if input_tokens else None
        self.blocks = nn.Sequential(
            *[BlockSelfAttention(f"block_{j}", d_model, d_ffd, n_heads, block_size, dropout, masked=True)
              for j in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_tokens)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def forward(self, x, targets=None):
        x = self.embedding(x) if self.input_tokens else x
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

        # if targets is None:
        #     loss = None
        # else:
        #     B, T, C = logits.shape
        #     logits = logits.view(B * T, C)
        #     targets = targets.view(B * T)
        #     loss = F.cross_entropy(logits, targets)
        #
        # return logits, loss
