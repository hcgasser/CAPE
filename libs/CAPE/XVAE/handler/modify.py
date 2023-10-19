import os
import pickle
from scipy import stats
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random

import torch
from torch.distributions import Categorical
from torch.nn import functional as F

import kit
import kit.globals as G
from kit import DD
from kit.maths import get_geometric_decay
from kit.log import log_info, log_warning
from kit.data import str_to_file
from kit.loch import seq_hashes_to_file
from kit.loch.path import get_fasta_file_path
from kit.loch.utils import get_seq_hash
from kit.bioinf import unpredicted_tokens
from kit.bioinf.fasta import seqs_to_fasta, fastas_to_seqs
from kit.bioinf.mhc import get_predictor
from kit.jobs import get_job_info
from kit.path import join

from CAPE.RL.reward import rewards_seqs
from CAPE.XVAE import load_model

output_folder = None

base_seq, base_seq_hash = None, None
MHC_Is = None
MHC_I_lengths = None
predictor_MHC_I = None
max_seq_length = None

rewards_settings = { 
    'rwd_impossible_token': -1,
    'rwd_premature_stop': None,
    'rwd_missing_stop': None
}

to_save = {
    'n_episodes': None,
    'Z_scale': None,
    'Zs': [],
    'seqs': [],
    'rewards': [],
    'unavailable': None
}

def run():
    if G.TASK.ID not in ['deimmunize', 'immunize']:
        raise ValueError(f"Unknown task: {G.TASK.ID}")
    modify(G.TASK.ID)


def modify(profile):
    global output_folder, base_seq, base_seq_hash, MHC_Is, MHC_I_lengths, predictor_MHC_I, max_seq_length, to_save

    modparams = DD.from_yaml(os.path.join(G.ENV.CONFIG, 'modparams', f'{G.TASK.MODPARAMS}.yaml'))
    N = modparams.N_SEQS
    n_episodes = modparams.N_EPISODES           # number of episodes to run
    to_save['n_episodes'] = n_episodes
    eval_episodes = modparams.EVAL.EPISODES     # eval every eval_episodes episodes
    alpha = modparams.ALPHA                     # RL step size
    temperature = modparams.TEMPERATURE.START   # temperature for sampling
    temperature_decay = get_geometric_decay(n_episodes, temperature, modparams.TEMPERATURE.END)  # decay of temperature

    log_info(f"Starting {G.TASK.ID} with {n_episodes} episodes, alpha={alpha}, temperature={temperature}, temperature_decay={temperature_decay:.3f}")

    # log task information to the output folder as well
    output_folder = join(load_model(G.TASK.MODEL_ID), profile, G.JOB.ID)
    G.TASK.to_yaml(join(output_folder, "TASK.yaml"))
    max_seq_length = G.DHPARAMS.MAX_SEQ_LEN

    # load sequence to deimmunize
    base_seq, base_seq_hash = get_base_seq()

    # load MHC-I alleles and the MHC-I predictor
    MHC_Is = G.TASK.MHCs.split('+')
    MHC_I_lengths = modparams.PREDICTOR_MHC_I.LENGTHS
    predictor_MHC_I = get_predictor(modparams.PREDICTOR_MHC_I.NAME)(folder=modparams.PREDICTOR_MHC_I.FOLDER, limit=modparams.PREDICTOR_MHC_I.LIMIT)

    # get the rewards of the base sequence
    R = rewards_seqs(predictor_MHC_I, [base_seq], MHC_Is, profile, MHC_I_lengths, **rewards_settings)
    log_info(f"Sum of base sequence rewards: {R.sum()} ")

    # get the original base sequence encoding in latent space
    Z_base, Z_scale = encode_seq(base_seq)
    W = torch.zeros_like(Z_base)  # initialize the trust region space around Z_base
    log_info(f"Norm Z_base: {torch.norm(Z_base).item():5.3f} Z_score: {torch.norm(Z_scale).item():5.3f}")

    # mark the base sequence and latents as to be saved
    to_save['Zs'].append(Z_base.cpu().numpy())
    to_save['seqs'].append(base_seq)
    to_save['Z_scale'] = Z_scale.cpu().numpy()

    episode_rewards = []
    
    # generate all episodes
    table_header = [f"{'Episode':<9} | {'Temp':>4} | {'Returns':^45} | {'Latent':^11} ",
                    f"{'':<9} | {'':>4} | {'Mean':>8}  {'Std':<8} {'Min':>8} {'Median':>8} {'Max':>8} | {'Norm':>5} {'Distance':>5} "]
    for episode in range(n_episodes):
        Z = Z_base + W_to_deltaZ(W, Z_scale, modparams.MAX_SIGMA)  # calculate the latent variable from the trust region
        if episode %  eval_episodes == 0:
            evaluate(Z, profile)
            log_info(table_header)

        # generate the actions for the period, one after the other and save the action gradients
        hidden, t_seqs, l_grads_w = None, None, []
        for t in range(max_seq_length):
            w = torch.stack([W.clone() for _ in range(N)], dim=0)
            w.requires_grad = True

            z = Z_base + W_to_deltaZ(w, Z_scale, modparams.MAX_SIGMA)

            # sample the next token
            seqs, t_seqs, t_probs, _, hidden = kit.model.generate(N, 1, initial_t_seqs=t_seqs, temperature=temperature, 
                                                          latents=z, hidden=hidden, 
                                                          unpredicted_tokens=unpredicted_tokens, 
                                                          max_batch_size=N, return_details=True)
            t_current_tokens = t_seqs[:, -1:]

            # save the gradients of the logarithmic policy w.r.t. the latent variable
            # for the current time-step and action chosen
            v = t_probs[:, -1, :].gather(-1, t_current_tokens).log().sum()  # extract the log likelihood of the actions chosen

            v.backward()

            grads_w = []
            for n in range(N):
                grads_w.append(w.grad[n].detach().clone())

            l_grads_w.append(torch.stack(grads_w))

            w.grad.zero_()


        # receive the rewards for the periode
        R = rewards_seqs(predictor_MHC_I, seqs, MHC_Is, profile, MHC_I_lengths, **rewards_settings)
        R_samples = R.sum(-1)

        mean_episode_reward = np.mean(R_samples)
        str_to_file(f"{mean_episode_reward:.1f}\n", join(output_folder, "rewards.txt"), True)
        episode_rewards.append(mean_episode_reward)

        t_grads_w = torch.stack(l_grads_w, dim=1)
        B = np.array([0] * (max_seq_length))  # baseline   
        for t in range(max_seq_length):
            sum_future_returns = R[:, t:].sum(axis=-1)  # sum the future rewards
            B[t] = sum_future_returns.mean()  # update Baseline
            for n in range(N):
                W = W + alpha * (sum_future_returns[n] - B[t]) * t_grads_w[n, t, :, :]

        log_info(f"{episode:<9} | {temperature:>4.2} | {np.mean(R_samples):>8.1f} \u00B1{np.std(R_samples):<8.1f} {np.min(R_samples):>8.1f} {np.median(R_samples):>8.1f} {np.max(R_samples):>8.1f} | {torch.norm(Z).item():>5.3f} {torch.norm(Z_base - Z).item():>5.3f} ")

        # update temperature
        temperature *= temperature_decay

    Z = Z_base + W_to_deltaZ(W, Z_scale, modparams.MAX_SIGMA)
    evaluate(Z, profile)

    slope, intercept, r_value, p_value, std_err = stats.linregress(range(n_episodes), np.array(episode_rewards))
    log_info(f"Slope: {slope}")

    predictor_MHC_I.save()
    save_results(profile, G.TASK.RESULTS.TO_LOCH)

    return slope


def evaluate(Z, profile):
    """Evaluate the sequence generated with the latent variable Z.
    
    Generates a sequence with the latent variable Z using greedy sampling.
    The rewards of this sequence are then calculated and the new information is saved.

    :param Z: torch.Tensor - the latent variable of the sequence
    """
    global predictor_MHC_I, MHC_Is, MHC_I_lengths, max_seq_length, output_folder, to_save

    # generate sequence
    seq = kit.model.generate(1, max_seq_length, temperature=0., latents=Z.unsqueeze(0), truncate_after_first_stop_token=True, unpredicted_tokens=unpredicted_tokens)[0]

    # obtain rewards of the sequence
    sum_rewards = rewards_seqs(predictor_MHC_I, [seq], MHC_Is, profile, MHC_I_lengths, **rewards_settings)[0].sum()

    chunk_size = 75
    text = [f"  Intermediate evaluation",
            f"    Generated sequence: "] \
            + ['      ' + seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)] \
            + [f"    Return: {sum_rewards:.2f}"]
    log_info(text)

    to_save['seqs'].append(seq)
    to_save['rewards'].append(sum_rewards)
    to_save['Zs'].append(Z.cpu().numpy())


def save_results(profile, to_loch):
    """Save the results of the deimmunization process.
    
    The results are saved in the output folder as a dictionary pickle file containing:
    - n_episodes: the number of episodes
    - Z_scale: the scale of the latent space
    - Zs: the latent variables of the sequences generated in each episode
    - seqs: the sequences generated in each episode
    - rewards: the rewards of the sequences generated in each episode
    - unavailable: the MHC-I peptides for which no prediction was available
    """

    global predictor_MHC_I, to_save

    seq_final = to_save['seqs'][-1]
    final_seq_hash = get_seq_hash(seq_final)
    seq_base = to_save['seqs'][0]
    base_seq_hash = get_seq_hash(seq_base)

    seqs_to_fasta([seq_final], join(G.ENV.ARTEFACTS, "generated", profile, f"{final_seq_hash}.fasta"))
    seqs_to_fasta([seq_base], join(G.ENV.ARTEFACTS, "generated", profile, f"{base_seq_hash}.fasta"))

    seq_hashes_to_file([final_seq_hash], os.path.join(G.ENV.ARTEFACTS, f'{G.DOMAIN}.CAPE-XVAE.{profile}.final.seq_hash'), append=True)
    seq_hashes_to_file([base_seq_hash], os.path.join(G.ENV.ARTEFACTS, f'{G.DOMAIN}.CAPE-XVAE.{profile}.base.seq_hash'), append=True)

    if to_loch:
        seqs_to_fasta([seq_final], get_fasta_file_path(final_seq_hash))
        seqs_to_fasta([seq_base], get_fasta_file_path(base_seq_hash))

    if len(predictor_MHC_I.unavailable) > 0:
        log_warning(f"Unavailabe MHC-I peptides: {len(predictor_MHC_I.unavailable)}")
        to_save['unavailable'] = dict(predictor_MHC_I.unavailable)

    with open(join(output_folder, "results.pickle"), "wb") as file:
        pickle.dump(to_save, file)


def get_base_seq():
    """Get the sequence to deimmunize.
    
    There are three possibiilities:
    - G.TASK.SEQ is a sequence: use this sequence
    - G.TASK.SEQ is "random": generate a random sequence using the decoder
    - G.TASK.SEQ is "natural": load a random natural sequence from the dataset

    :return: base_seq - the sequence to deimmunize
    :return: base_seq_hash - the hash of the sequence
    """

    if G.TASK.SEQ == "random":  # generate a random sequence using the decoder
        base_seq = kit.model.generate(1, max_seq_length)[0]
        base_seq_hash = get_seq_hash(base_seq)
        seqs_to_fasta([base_seq], join(G.ENV.ARTEFACTS, "sequences", f'{base_seq_hash}.fasta'))
        G.TASK.SEQ = f"random:{base_seq_hash}"
    elif G.TASK.SEQ == "natural": # load a random natural sequence from the dataset
        natural_seqs = []
        for split in ['TRAIN', 'VAL', 'TEST']:
            split_file_path = join(G.ENV.INPUT, G.DHPARAMS.FOLDER, f'{split}.fasta')
            natural_seqs += fastas_to_seqs(split_file_path)
        base_seq = random.choice(natural_seqs)
        base_seq_hash = get_seq_hash(base_seq)
        G.TASK.SEQ = f"natural:{base_seq_hash}"
    else:  # use the sequence provided
        base_seq = G.TASK.SEQ
        base_seq_hash = get_seq_hash(base_seq)

    return base_seq, base_seq_hash


def encode_seq(seq):
    """Encode a sequence in the latent space.
    
    :param seq: str - the sequence to encode
    :return: Z - the latent variable of the sequence
    :return: Z_scale - the scale of the latent space"""

    t_seq, mask_input = kit.bioinf.seqs_to_tensor([seq], start="", end="", device=kit.device)

    Z, p, q = kit.model.encode(t_seq)
    Z = Z.detach().squeeze(0)  # squeeze out batch dimension
    Z_scale = q.scale.detach().squeeze(0)  # squeeze out batch dimension

    return Z, Z_scale


def W_to_deltaZ(W, Z_scale, max_sigma):
    """Calculate the latent variable from the trust region vector W
    
    :param W: torch.Tensor - the trust region vector
    :param Z_scale: torch.Tensor - the scale of the latent space
    :param max_sigma: float - the maximum sigma to use"""

    return torch.tanh(W) * Z_scale * max_sigma
