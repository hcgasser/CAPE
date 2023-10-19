import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tempfile

from kit.log import log_info, log_caller
from kit.nn import concatenate_nested, slice_nested
from kit.nn.encodings import LearnedTokenEmbedding, AAEmbedding, PositionalEncoding
from kit.nn.transformer import TransformerLM, EncoderStack, DecoderStack, BlockCrossAttention
from kit.bioinf import aa_1_to_idx, seqs_to_tensor, tensor_to_seqs
from kit.path import join

from .Basic import CapeVAE


class CapeTransformerXEncoder(nn.Module):
    def __init__(self, n_latents, d_latent, d_model, d_ffd, n_heads, n_layers, block_size, dropout, n_tokens):
        super().__init__()
        self.n_latents = n_latents
        self.d_model = d_model

        self.embedding = LearnedTokenEmbedding(block_size, n_tokens, d_model)
        self.self_encoder = EncoderStack(d_model, d_ffd, n_heads, n_layers, block_size, dropout)
        self.cross_attention = BlockCrossAttention("block_XA", d_model, d_ffd, n_heads, block_size, dropout, masked_sa=False, masked_xa=False)
        self.cross_attention_params = torch.nn.Parameter(torch.zeros(n_latents, d_model))
        self.ln = nn.LayerNorm(d_model)
        self.mu = nn.Linear(d_model, d_latent)
        self.log_var = nn.Linear(d_model, d_latent)

    def forward(self, x, stochastic=True):
        B, T = x.shape
        x = self.embedding(x)
        x = self.self_encoder(x)
        x, _ = self.cross_attention(self.cross_attention_params.expand(B, self.n_latents, self.d_model), x)
        x = self.ln(x)
        x = F.leaky_relu(x)
        # return x
        mu = self.mu(x)
        if stochastic:
            std = torch.exp(self.log_var(x) / 2)
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
        else:
            z, p, q = mu, None, None
        return z, p, q


class CapeTransformerXDecoder(nn.Module):
    def __init__(self, d_latent, d_model, d_ffd, n_heads, n_layers, block_size, dropout, n_tokens):
        super().__init__()
        self.embedding = LearnedTokenEmbedding(block_size, n_tokens, d_model)
        self.upsample = nn.Linear(d_latent, d_model)
        self.decoder = DecoderStack(d_model, d_ffd, n_heads, n_layers, block_size, dropout)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_tokens)

    def forward(self, x=None, x_kv=None, positions=None, n_positions=None, upsample=True, hidden=None):
        x_kv = self.upsample(x_kv) if (upsample and x_kv is not None) else x_kv
        x = self.embedding(x=x, positions=positions, n_positions=n_positions)
        x, hidden = self.decoder(x, x_kv, hidden=hidden)
        x = self.ln(x)
        x = F.leaky_relu(x)
        x = self.head(x)
        return x, hidden


class CapeTransformerXVAE(CapeVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log_caller(self)

        self.ar = 1
        self.mu_hat = torch.nn.Parameter(torch.zeros(self.hparams.N_LATENTS, self.hparams.D_LATENT))
        self.cov_hat = torch.nn.Parameter(torch.stack(
            [torch.diag(torch.ones(self.hparams.D_LATENT)) for _ in range(self.hparams.N_LATENTS)])
        )
        self.DECODER_INPUT = self.hparams.DECODER_INPUT if "DECODER_INPUT" in self.hparams else "SEQ"
        self.decoder_input_latent, self.decoder_input_seq, self.decoder_input_n_positions = False, False, False
        for dec_input in self.DECODER_INPUT:
            if dec_input == "Z":
                self.decoder_input_latent = True
            if dec_input == "X":
                self.decoder_input_seq = True
            if dec_input == "N":
                self.decoder_input_n_positions = True

        log_info(f"Decoder input - Latent: {self.decoder_input_latent}, Sequence: {self.decoder_input_seq}, Next position: {self.decoder_input_n_positions}")

        self.permutate = self.hparams.PERMUTATE if "PERMUTATE" in self.hparams else False

        self.encoder = CapeTransformerXEncoder(self.hparams.N_LATENTS, self.hparams.D_LATENT,
            self.hparams.D_MODEL, self.hparams.D_FFD, self.hparams.N_HEADS, self.hparams.N_LAYERS,
            self.hparams.BLOCK_SIZE, self.hparams.DROPOUT, self.hparams.N_TOKENS)

        self.decoder = CapeTransformerXDecoder(self.hparams.D_LATENT, self.hparams.D_MODEL,
            self.hparams.D_FFD, self.hparams.N_HEADS, self.hparams.N_LAYERS, self.hparams.BLOCK_SIZE,
            self.hparams.DROPOUT, self.hparams.N_TOKENS)
        self.temp_decoder_weights_file_path = None

    def forward(self, x, stochastic=True, hidden=None, n_positions=None):
        """ feeds an input sequence x ("*MGGKWSK*") through the encoder to obtain a latent representation z.
        This is then fed together with
        - the position of the token (starting * is 0 and first residue is 1)
        - optional: the original sequence x
        - optional: the next position to be predicted
        into the decoder.

        The function returns the predicted next tokens (y_hat) in a 1 to L order, together with the latent representation (z),
        the target latent distribution (p), the actual latent distributions (q) and the decoder hidden variables
        """

        B, L = x.shape

        z_, p, q = self.encoder(x, stochastic)
        # z = self.encoder(x, stochastic)

        positions = torch.arange(0, L-self.ar, device=self.device).expand(B, L-self.ar)
        if self.decoder_input_n_positions:
            if n_positions is None:
                if self.permutate:
                    n_positions = torch.stack([torch.randperm(L-self.ar, device=self.device) + 1 for _ in range(B)], dim=0)
                    # n_positions = torch.randperm(L-self.ar, device=self.device) + 1
                    positions = torch.concat(
                        [torch.zeros((B, 1), device=self.device, dtype=n_positions.dtype), n_positions[:, :-1]],
                        dim=-1)
                else:
                    n_positions = positions + 1

        x = x.gather(-1, positions) if self.decoder_input_seq else None
        z = z_ if self.decoder_input_latent else None

        y_hat_, hidden = self.decoder(x=x, x_kv=z, positions=positions, n_positions=n_positions, hidden=hidden)
        if self.permutate:
            reverse = torch.stack([torch.argsort(n_positions[j]) for j in range(B)], dim=0)
            y_hat = y_hat_.gather(1, reverse.unsqueeze(-1).expand(B, L-self.ar, self.hparams.N_TOKENS))  # reverse permutation
            # y_hat = y_hat[:, torch.argsort(n_positions)]
        else:
            y_hat = y_hat_

        # print(f"{self.encoder.cross_attention_params.max().item()} {self.encoder.cross_attention_params.min().item()}")

        return y_hat, z_, p, q, hidden

    def save_decoder(self):
        log_caller(self)
        self.temp_decoder_weights_file_path = tempfile.mktemp()
        torch.save(self.decoder.state_dict(), self.temp_decoder_weights_file_path)

    def reset_decoder(self):
        log_caller(self)
        decoder = torch.load(self.temp_decoder_weights_file_path)
        # torch.save(self.decoder.state_dict(), self.temp_decoder_weights_file_path)
        self.decoder.load_state_dict(decoder)

    def encode(self, idx, stochastic=True):
        z, p, q = self.encoder(idx, stochastic)
        return z, p, q

    def sample_latents(self, N):
        ds = [torch.distributions.multivariate_normal.MultivariateNormal(mu, cov)
              for mu, cov in zip(self.mu_hat, self.cov_hat)]
        latents = torch.stack([
            torch.stack([d.rsample() for d in ds], dim=0)
            for _ in range(N)])
        return latents


    def generate(self, n_seqs, max_new_tokens, 
                temperature=0., 
                initial_seqs=None, initial_t_seqs=None, 
                latents=None, hidden=None, 
                truncate_after_first_stop_token=False, unpredicted_tokens="", 
                max_batch_size=None, return_details=False):
        t_seqs, t_probs, last_hidden, t_latents = None, None, None, None

        max_batch_size = self.hparams.BATCH_SIZE if max_batch_size is None else max_batch_size

        self.eval()

        if initial_t_seqs is None:
            if initial_seqs is None:
                initial_seqs = ['*']*n_seqs
            else:
                initial_seqs = [('*' + seq) if (len(seq) == 0 or seq[0] != '*') else seq for seq in initial_seqs]
                if not all([len(seq) == len(initial_seqs[0]) for seq in initial_seqs]):
                    raise ValueError("All initial sequences must have the same length (start tokens get automatically added))")

        n_generated_seqs = 0
        while n_generated_seqs < n_seqs:
            # get the batch_size for the next iteration
            batch_size = np.min([n_seqs, n_seqs - n_generated_seqs, max_batch_size])
            
            # the initial sequence is the start token
            if initial_t_seqs is None:
                batch_t_seqs, _ = seqs_to_tensor(initial_seqs[n_generated_seqs:n_generated_seqs + batch_size], start="", end="", device=self.device)
            else:
                batch_t_seqs = initial_t_seqs[n_generated_seqs:n_generated_seqs + batch_size]

            # sample the latent variables
            batch_t_latents = self.sample_latents(batch_size) if latents is None else latents[n_generated_seqs:n_generated_seqs + batch_size]
            batch_t_latents_up = self.decoder.upsample(batch_t_latents)

            batch_current_tokens = batch_t_seqs[:, -1:]  # the current tokens are the last tokens in the sequence
            # generate the next tokens
            batch_t_probs = None

            if hidden is None:
                batch_current_hidden = None
            else:
                batch_current_hidden = slice_nested(hidden, n_generated_seqs, n_generated_seqs + batch_size)

            for _ in range(max_new_tokens):
                # determine the current position in the sequence and the next to predict
                current_pos = torch.tensor([batch_t_seqs.shape[-1] - 1]).to(self.device)
                next_pos = (current_pos + 1) if self.decoder_input_n_positions else None

                # if the decoder does not receive the current token, then set it to None
                if not self.decoder_input_seq:
                    batch_current_tokens = None

                y, batch_current_hidden = self.decoder(x=batch_current_tokens, x_kv=batch_t_latents_up, positions=current_pos, n_positions=next_pos, upsample=False, hidden=batch_current_hidden)

                for token in unpredicted_tokens:
                    y[:, -1, aa_1_to_idx[token]] = float('-inf')

                # update the current tokens
                p = None
                if temperature == 0.:
                    batch_current_tokens = y.max(dim=-1).indices
                else:
                    y = y/temperature
                    p = F.softmax(y, dim=-1)
                    batch_current_tokens = torch.multinomial(p[:, -1, :], num_samples=1, replacement=True)

                batch_t_seqs = torch.concat([batch_t_seqs, batch_current_tokens], dim=-1)
                batch_t_probs = p if batch_t_probs is None else torch.concat([batch_t_probs, p], dim=-2)

            batch_last_hidden = batch_current_hidden

            # add the generated data to the existing data
            t_seqs = batch_t_seqs if t_seqs is None else torch.concat([t_seqs, batch_t_seqs], dim=0)
            t_probs = batch_t_probs if t_probs is None else torch.concat([t_probs, batch_t_probs], dim=0)
            t_latents = batch_t_latents if t_latents is None else torch.concat([t_latents, batch_t_latents], dim=0)
            last_hidden = batch_last_hidden if last_hidden is None else concatenate_nested(last_hidden, batch_last_hidden, dim=0)
            n_generated_seqs = t_seqs.shape[0]

        seqs = tensor_to_seqs(t_seqs[:,1:])
        # if wanted, remove all tokens after the first stop token
        if truncate_after_first_stop_token:
            for i, seq in enumerate(seqs):
                if '*' in seq:
                    seqs[i] = seq[:seq.index('*') + 1]

        if return_details:
            return seqs, t_seqs, t_probs, t_latents, last_hidden
        return seqs
        