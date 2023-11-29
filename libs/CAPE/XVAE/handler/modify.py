""" This module contains the code for the modification of a sequence using the XVAE model.
It is the handler for the modify task.
"""

import os
import pickle
import random

import numpy as np

import torch

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
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.path import join

from CAPE.RL.reward import rewards_seqs
from CAPE.XVAE import load_ckpt, add_shortcut_to_artefacts
from CAPE.profiles import Profile

TO_SAVE = {
    "n_episodes": None,
    "Z_scale": None,
    "Zs": [],
    "seqs": [],
    "rewards": [],
    "unavailable": None,
}


def run():
    """main entrance into the handler"""

    profile = Profile(G.TASK.PROFILE)
    modify(profile)


def modify(profile):
    """Modify a sequence using the XVAE model.

    :param profile: Profile - the profile to use for the modification
    """

    global TO_SAVE

    modparams = DD.from_yaml(
        os.path.join(G.ENV.CONFIG, "modparams", f"{G.TASK.MODPARAMS}.yaml")
    )
    N = modparams.N_SEQS
    n_episodes = modparams.N_EPISODES  # number of episodes to run
    TO_SAVE["n_episodes"] = n_episodes
    eval_episodes = modparams.EVAL.EPISODES  # eval every eval_episodes episodes
    alpha = modparams.ALPHA  # RL step size
    temperature = modparams.TEMPERATURE.START  # temperature for sampling
    temperature_decay = get_geometric_decay(
        n_episodes, temperature, modparams.TEMPERATURE.END
    )  # decay of temperature

    log_info(
        f"Starting {profile} with {n_episodes} episodes, alpha={alpha}, "
        f"temperature={temperature}, temperature_decay={temperature_decay:.3f}"
    )

    # log task information to the output folder as well
    ckpt_dir_path = load_ckpt(G.TASK.CKPT_ID)
    job_output_dir_path = get_job_output_dir_path(ckpt_dir_path, profile)
    G.TASK.to_yaml(join(job_output_dir_path, "TASK.yaml"))
    max_seq_length = G.DHPARAMS.MAX_SEQ_LEN

    # load sequence to modify
    base_seq, _ = get_base_seq()

    # load MHC-I alleles and the MHC-I predictor
    mhc_1_alleles = G.TASK.MHCs.split("+")
    mhc_1_peptide_lengths = modparams.PREDICTOR_MHC_I.LENGTHS
    predictor_mhc_1 = Mhc1Predictor.get_predictor(modparams.PREDICTOR_MHC_I.NAME)(
        data_dir_path=modparams.PREDICTOR_MHC_I.FOLDER,
        limit=modparams.PREDICTOR_MHC_I.LIMIT,
    )

    # get the rewards of the base sequence
    R = rewards_seqs(
        predictor_mhc_1, [base_seq], mhc_1_alleles, profile, mhc_1_peptide_lengths
    )
    log_info(f"Sum of base sequence rewards: {R.sum()} ")

    # get the original base sequence encoding in latent space
    Z_base, Z_scale = encode_seq(base_seq, modparams.SEQ_ENC_STOCHASTIC)
    W = torch.zeros_like(Z_base)  # initialize the trust region space around Z_base
    log_info(
        f"Norm Z_base: {torch.norm(Z_base).item():5.3f} Z_score: {torch.norm(Z_scale).item():5.3f}"
    )

    # mark the base sequence and latents as to be saved
    TO_SAVE["Zs"].append(Z_base.cpu().numpy())
    TO_SAVE["seqs"].append(base_seq)
    TO_SAVE["Z_scale"] = Z_scale.cpu().numpy()

    # generate all episodes
    line_1 = f"{'Episode':<9} | {'Temp':>4} | {'Returns':^45} | {'Latent':^11} "
    line_2 = (
        f"{'':<9} | {'':>4} | "
        f"{'Mean':>8}  {'Std':<8} {'Min':>8} {'Median':>8} {'Max':>8} | "
        f"{'Norm':>5} {'Distance':>5} "
    )
    table_header = [line_1, line_2]

    for episode in range(n_episodes):
        Z = Z_base + transform_auxiliary_to_delta_latent(
            W, Z_scale, modparams.MAX_SIGMA
        )  # calculate the latent variable from the trust region
        if episode % eval_episodes == 0:
            evaluate(Z, profile, predictor_mhc_1, mhc_1_alleles, mhc_1_peptide_lengths)
            log_info(table_header)

        # generate the actions for the period, one after the other and save the action gradients
        hidden, t_seqs, l_grads_w = None, None, []
        for t in range(max_seq_length):
            w = torch.stack([W.clone() for _ in range(N)], dim=0)
            w.requires_grad = True

            z = Z_base + transform_auxiliary_to_delta_latent(
                w, Z_scale, modparams.MAX_SIGMA
            )

            # sample the next token
            seqs, t_seqs, t_probs, _, hidden = kit.MODEL.generate(
                N,
                1,
                initial_t_seqs=t_seqs,
                temperature=temperature,
                latents=z,
                hidden=hidden,
                unpredicted_tokens=unpredicted_tokens,
                max_batch_size=N,
                return_details=True,
            )
            t_current_tokens = t_seqs[:, -1:]

            # save the gradients of the logarithmic policy w.r.t. the latent variable
            # for the current time-step and action chosen
            v = (
                t_probs[:, -1, :].gather(-1, t_current_tokens).log().sum()
            )  # extract the log likelihood of the actions chosen

            v.backward()

            grads_w = []
            for n in range(N):
                grads_w.append(w.grad[n].detach().clone())

            l_grads_w.append(torch.stack(grads_w))

            w.grad.zero_()

        # receive the rewards for the periode
        R = rewards_seqs(
            predictor_mhc_1, seqs, mhc_1_alleles, profile, mhc_1_peptide_lengths
        )

        t_grads_w = torch.stack(l_grads_w, dim=1)
        B = np.array([0] * (max_seq_length))  # baseline
        for t in range(max_seq_length):
            sum_future_returns = R[:, t:].sum(axis=-1)  # sum the future rewards
            B[t] = sum_future_returns.mean()  # update Baseline
            for n in range(N):
                W = W + alpha * (sum_future_returns[n] - B[t]) * t_grads_w[n, t, :, :]

        R_samples = R.sum(-1)
        r_mean = np.mean(R_samples)
        r_std = np.std(R_samples)
        r_min = np.min(R_samples)
        r_median = np.median(R_samples)
        r_max = np.max(R_samples)
        z_norm = torch.norm(Z).item()
        z_dist = torch.norm(Z_base - Z).item()

        log_info(
            f"{episode:<9} | {temperature:>4.2} | "
            f"{r_mean:>8.1f} \u00B1{r_std:<8.1f} "
            f"{r_min:>8.1f} "
            f"{r_median:>8.1f} "
            f"{r_max:>8.1f} | "
            f"{z_norm:>5.3f} {z_dist:>5.3f} "
        )

        str_to_file(f"{r_mean:.1f}\n", join(job_output_dir_path, "rewards.txt"), True)

        # update temperature
        temperature *= temperature_decay

    Z = Z_base + transform_auxiliary_to_delta_latent(W, Z_scale, modparams.MAX_SIGMA)
    evaluate(Z, profile, predictor_mhc_1, mhc_1_alleles, mhc_1_peptide_lengths)

    predictor_mhc_1.save()
    save_results(profile, G.TASK.RESULTS.TO_LOCH, ckpt_dir_path, predictor_mhc_1)


def evaluate(Z, profile, predictor_mhc_1, mhc_1_alleles, mhc_1_peptide_lengths):
    """Evaluate the sequence generated with the latent variable Z.

    Generates a sequence with the latent variable Z using greedy sampling.
    The rewards of this sequence are then calculated and the new information is saved.

    :param Z: torch.Tensor - the latent variable of the sequence
    """
    global TO_SAVE

    max_seq_length = G.DHPARAMS.MAX_SEQ_LEN

    # generate sequence
    seq = kit.MODEL.generate(
        1,
        max_seq_length,
        temperature=0.0,
        latents=Z.unsqueeze(0),
        truncate_after_first_stop_token=True,
        unpredicted_tokens=unpredicted_tokens,
    )[0]

    # obtain rewards of the sequence
    rewards, rewards_natural, rewards_artificial, _ = rewards_seqs(
        predictor_mhc_1,
        [seq],
        mhc_1_alleles,
        profile,
        mhc_1_peptide_lengths,
        return_details=True,
    )
    s_rwd = rewards[0].sum()
    s_rwd_nat = rewards_natural[0].sum()
    s_rwd_art = rewards_artificial[0].sum()

    chunk_size = 75
    text = (
        ["  Intermediate evaluation", "    Generated sequence: "]
        + ["      " + seq[i : i + chunk_size] for i in range(0, len(seq), chunk_size)]
        + [
            f"    Return: {s_rwd:.2f} (natural: {s_rwd_nat:.2f}, artificial: {s_rwd_art:.2f})"
        ]
    )
    log_info(text)

    TO_SAVE["seqs"].append(seq)
    TO_SAVE["rewards"].append(s_rwd)
    TO_SAVE["Zs"].append(Z.cpu().numpy())


def save_results(profile, to_loch, ckpt_dir_path, predictor_mhc_1):
    """Save the results of the modification process.

    The results are saved in the output folder as a dictionary pickle file containing:
    - n_episodes: the number of episodes
    - Z_scale: the scale of the latent space
    - Zs: the latent variables of the sequences generated in each episode
    - seqs: the sequences generated in each episode
    - rewards: the rewards of the sequences generated in each episode
    - unavailable: the MHC-I peptides for which no prediction was available
    """

    global TO_SAVE

    job_output_dir_path = get_job_output_dir_path(ckpt_dir_path, profile)

    seq_final = TO_SAVE["seqs"][-1]
    final_seq_hash = get_seq_hash(seq_final)
    seq_base = TO_SAVE["seqs"][0]
    base_seq_hash = get_seq_hash(seq_base)

    seqs_to_fasta(
        [seq_final],
        join(ckpt_dir_path, "generated", str(profile), f"{final_seq_hash}.fasta"),
    )
    seqs_to_fasta(
        [seq_base],
        join(
            ckpt_dir_path, "generated", str(profile), "base", f"{base_seq_hash}.fasta"
        ),
    )
    add_shortcut_to_artefacts("generated", ckpt_dir_path, G.TASK.CKPT_ID)

    seq_hash_final_file_path = (
        os.path.join(G.ENV.ARTEFACTS, f"{G.DOMAIN}.CAPE-XVAE.{profile}.final.seq_hash")
        if G.TASK.RESULTS.SEQ_HASH_FINAL_FILE_PATH is None
        else G.TASK.RESULTS.SEQ_HASH_FINAL_FILE_PATH
    )
    seq_hashes_to_file([final_seq_hash], seq_hash_final_file_path, append=True)
    seq_hash_base_file_path = (
        os.path.join(G.ENV.ARTEFACTS, f"{G.DOMAIN}.CAPE-XVAE.{profile}.base.seq_hash")
        if G.TASK.RESULTS.SEQ_HASH_BASE_FILE_PATH is None
        else G.TASK.RESULTS.SEQ_HASH_BASE_FILE_PATH
    )
    seq_hashes_to_file([base_seq_hash], seq_hash_base_file_path, append=True)

    if to_loch:
        removed_tokens = "*-" + unpredicted_tokens
        seqs_to_fasta(
            [seq_final],
            get_fasta_file_path(final_seq_hash),
            translate=("", "", removed_tokens),
        )
        seqs_to_fasta(
            [seq_base],
            get_fasta_file_path(base_seq_hash),
            translate=("", "", removed_tokens),
        )

    if len(predictor_mhc_1.unavailable) > 0:
        log_warning(f"Unavailabe MHC-I peptides: {len(predictor_mhc_1.unavailable)}")
        TO_SAVE["unavailable"] = dict(predictor_mhc_1.unavailable)

    with open(join(job_output_dir_path, "results.pickle"), "wb") as file:
        pickle.dump(TO_SAVE, file)


def get_base_seq():
    """Get the sequence to modify.

    There are three possibiilities:
    - G.TASK.SEQ is a sequence: use this sequence
    - G.TASK.SEQ is "random": generate a random sequence using the decoder
    - G.TASK.SEQ is "natural": load a random natural sequence from the dataset

    :return: base_seq - the sequence to modify
    :return: base_seq_hash - the hash of the sequence
    """

    max_seq_length = G.DHPARAMS.MAX_SEQ_LEN

    if G.TASK.SEQ == "random":  # generate a random sequence using the decoder
        log_info("Generating random sequence")
        base_seq = kit.MODEL.generate(1, max_seq_length)[0]
        base_seq_hash = get_seq_hash(base_seq)
        seqs_to_fasta(
            [base_seq], join(G.ENV.ARTEFACTS, "sequences", f"{base_seq_hash}.fasta")
        )
        G.TASK.SEQ = f"random:{base_seq_hash}"
    elif G.TASK.SEQ == "natural":  # load a random natural sequence from the dataset
        log_info("Loading random natural sequence")
        natural_seqs = []
        for split in ["TRAIN", "VAL", "TEST"]:
            split_file_path = join(G.ENV.INPUT, G.DHPARAMS.FOLDER, f"{split}.fasta")
            natural_seqs += fastas_to_seqs(split_file_path)
        base_seq = random.choice(natural_seqs)
        base_seq_hash = get_seq_hash(base_seq)
        G.TASK.SEQ = f"natural:{base_seq_hash}"
    else:  # use the sequence provided
        log_info("Using provided sequence")
        base_seq = G.TASK.SEQ
        base_seq_hash = get_seq_hash(base_seq)

    return base_seq, base_seq_hash


def encode_seq(seq, stochastic):
    """Encode a sequence in the latent space.

    :param seq: str - the sequence to encode
    :return: Z - the latent variable of the sequence
    :return: Z_scale - the scale of the latent space"""

    t_seq, _ = kit.bioinf.seqs_to_tensor([seq], start="", end="", device=kit.DEVICE)

    Z, _, q = kit.MODEL.encode(t_seq, stochastic=stochastic)
    Z = Z.detach().squeeze(0)  # squeeze out batch dimension
    Z_scale = q.scale.detach().squeeze(0)  # squeeze out batch dimension

    return Z, Z_scale


def get_job_output_dir_path(ckpt_dir_path, profile):
    return join(ckpt_dir_path, "jobs", "modify", str(profile), G.JOB.ID)


def transform_auxiliary_to_delta_latent(W, Z_scale, max_sigma):
    """Calculate the latent variable from the trust region vector W

    :param W: torch.Tensor - the trust region vector
    :param Z_scale: torch.Tensor - the scale of the latent space
    :param max_sigma: float - the maximum sigma to use"""

    return torch.tanh(W) * Z_scale * max_sigma
