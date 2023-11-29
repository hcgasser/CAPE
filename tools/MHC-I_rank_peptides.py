#!/usr/bin/env python

import os
import argparse
import sys
import traceback
import pdb
import copy

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from kit.path import join
from kit.log import setup_logger
from kit.bioinf import generate_random_aa_seq, AA1_STD, N_AA_STD
from kit.bioinf.immuno.mhc_1 import Mhc1Predictor
from kit.data import str_to_file, file_to_str


def main(_args):
    lengths = [int(length) for length in _args.lengths.split("+")]
    alleles = _args.alleles.split("+")
    peptides_per_length = _args.peptides_per_length
    backend = _args.backend
    tasks = _args.tasks.split("+")
    percentiles = [99.9, 99.5, 99.0, 98.0, 97.5, 95.0, 90.0, 75.0, 50.0]
    limit_rank = 0.02

    if _args.output != "":
        folder = join(_args.output)
    else:
        folder = join(
            os.environ["DATA"],
            "processed",
            "MHC_class_I",
            "random_peptides",
            _args.lengths,
            f"random_{int(peptides_per_length)}",
            backend,
        )

    finished_alleles_file = os.path.join(folder, "finished_alleles.txt")
    finished_alleles = file_to_str(finished_alleles_file).split("\n")
    alleles = [allele for allele in alleles if allele not in finished_alleles]

    random_peptides_file = os.path.join(folder, "random_peptides.txt")

    if os.path.exists(random_peptides_file):
        print("Read existing random_peptides from disk")
        random_peptides = file_to_str(random_peptides_file).split("\n")
    else:
        print("Generate new random_peptides")
        random_peptides = []
        for length in tqdm(lengths, "Generate random peptides"):
            random_peptides += [
                generate_random_aa_seq(length) for _ in range(peptides_per_length)
            ]
        str_to_file("\n".join(random_peptides), random_peptides_file)

    df_ranks = pd.DataFrame(index=random_peptides, columns=["length"])
    df_ranks.index.name = "peptide"
    df_ranks["length"] = df_ranks.apply(lambda row: len(row.name), axis=1)

    predictor = None
    PWMs = {}
    PWMs_log = {}

    pbar = tqdm(alleles, "Alleles")
    for allele in pbar:
        allele_for_path = allele.replace("*", "_")
        if "rank" in tasks:
            filename = join(folder, f"{allele_for_path}.csv")
            selection = "y"
            if os.path.exists(filename):
                selection = input(
                    f"{filename} already exists. Would you like to overwrite? (y/n)"
                )

            if selection == "y":
                pbar.set_description(f"{allele} - rank")
                predictor = (
                    Mhc1Predictor.get_predictor(backend)()
                    if predictor is None
                    else predictor
                )
                for peptide in random_peptides:
                    predictor.peptide_rank(
                        peptide, allele, queue_if_missing=True, predict_if_missing=False
                    )
                predictor.predict_missing_peptides()
                df_ranks[allele] = df_ranks.apply(
                    lambda row: predictor.peptide_rank(row.name, allele), axis=1
                )
                df_ranks[[allele]].to_csv(filename)

        if "pwm" in tasks:
            pbar.set_description(f"{allele} - pwm")
            if allele not in df_ranks:
                df = pd.read_csv(join(folder, f"{allele_for_path}.csv")).set_index(
                    "peptide"
                )
                df_ranks = df_ranks.join(df, how="left")

            PWMs[allele] = {}
            for length in lengths:
                pwm = np.zeros((N_AA_STD, length))
                PWMs[allele][length] = pd.DataFrame(
                    columns=list(range(length)),
                    index=list(AA1_STD),
                    data=pwm,
                )
                PWMs[allele][length].index.name = "AA"

            # count how often each AA is observed at each position
            for peptide, row in tqdm(
                df_ranks.iterrows(), "count AAs at positions", leave=False
            ):
                length = len(peptide)
                for pos, aa in enumerate(peptide):
                    if row[allele] <= limit_rank:
                        PWMs[allele][length].loc[aa, pos] += 1

            # convert counts to probs and log probs
            PWMs_log[allele] = {}
            for length in tqdm(lengths, "generate PWMs for each length", leave=False):
                PWMs[allele][length] /= PWMs[allele][length].sum(axis=0)
                pwm_log = copy.deepcopy(PWMs[allele][length])
                np.seterr(divide="ignore")
                pwm_log.loc[:, :] = np.nan_to_num(np.log(pwm_log), neginf=-1e9)
                np.seterr(divide="warn")
                PWMs_log[allele][length] = pwm_log

                # save the produced PWMs
                PWMs[allele][length].to_csv(
                    join(
                        folder,
                        "pwm",
                        allele_for_path,
                        f"{allele_for_path}-{length}.csv",
                    )
                )
                PWMs_log[allele][length].to_csv(
                    join(
                        folder,
                        "pwm",
                        allele_for_path,
                        f"{allele_for_path}-{length}_log.csv",
                    )
                )

            # score all random peptides with the log pwm
            allele_scores = {"peptide": [], f"{allele}_score": []}
            for peptide, row in tqdm(
                df_ranks.iterrows(), "score random peptides with PWM", leave=False
            ):
                length = len(peptide)
                score = 0.0
                for pos, aa in enumerate(peptide):
                    score += PWMs_log[allele][length].loc[aa, pos]
                allele_scores["peptide"].append(peptide)
                allele_scores[f"{allele}_score"].append(score)

            df_ranks = df_ranks.join(pd.DataFrame(allele_scores).set_index("peptide"))

            # generate percentiles
            for length in lengths:
                df_percentiles = pd.DataFrame(
                    index=[f"pc_{p}" for p in percentiles]
                    + [f"recall_{p}" for p in percentiles]
                    + [f"precision_{p}" for p in percentiles]
                    + [f"F1_{p}" for p in percentiles],
                    columns=[allele],
                )
                df_percentiles.index.name = "Info"
                df_percentiles.loc[
                    [f"pc_{p}" for p in percentiles], allele
                ] = np.percentile(
                    df_ranks.query(f"length == {length}")[f"{allele}_score"],
                    percentiles,
                )

                for p in percentiles:
                    score = df_percentiles.loc[f"pc_{p}", allele]
                    true_positives = df_ranks.query(
                        f"`{allele}_score` > {score} "
                        f"and `{allele}` <= {limit_rank} "
                        f"and length == {length} "
                    ).shape[0]
                    false_negatives = df_ranks.query(
                        f"`{allele}_score` <= {score} "
                        f"and `{allele}` <= {limit_rank} "
                        f"and length == {length} "
                    ).shape[0]
                    false_positives = df_ranks.query(
                        f"`{allele}_score` > {score} "
                        f"and `{allele}` > {limit_rank} "
                        f"and length == {length} "
                    ).shape[0]

                    df_percentiles.loc[f"recall_{p}", allele] = (
                        true_positives / (true_positives + false_negatives)
                        if (true_positives + false_negatives) > 0
                        else None
                    )
                    df_percentiles.loc[f"precision_{p}", allele] = (
                        true_positives / (true_positives + false_positives)
                        if (true_positives + false_positives) > 0
                        else None
                    )
                    df_percentiles.loc[f"F1_{p}", allele] = (
                        true_positives
                        / (true_positives + 0.5 * (false_positives + false_negatives))
                        if (true_positives + 0.5 * (false_positives + false_negatives))
                        > 0
                        else None
                    )

                df_percentiles.to_csv(
                    os.path.join(
                        folder,
                        "pwm",
                        allele_for_path,
                        f"pc-{allele.replace('*', '_')}-{limit_rank}-{length}.csv",
                    )
                )
            str_to_file(f"{allele}\n", finished_alleles_file, append=True)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--lengths",
        type=str,
        default="8+9+10",
        help="the lengths of peptides to evaluate (e.g. 8+9+10)",
    )
    argparser.add_argument(
        "--peptides_per_length",
        type=int,
        default=100,
        help="the number of peptides per length to evaluate (e.g. 100000)",
    )
    argparser.add_argument(
        "--alleles",
        type=str,
        default="",
        help=(
            "the MHC alleles to evaluate the random peptides on "
            "(e.g. HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01)"
        ),
    )
    argparser.add_argument(
        "--backend",
        type=str,
        default="Mhc1PredictorNetMhcPan",
        help="the backend predictor to use (e.g. netMHCpan or MHCflurry)",
    )
    argparser.add_argument(
        "--tasks",
        type=str,
        default="rank+pwm",
        help=(
            "rank uses the backend to rank the peptides, "
            "pwm loads previously ranked peptides and produces position weight matrices"
        ),
    )
    argparser.add_argument(
        "--output", type=str, default="", help="the directory path for the output"
    )
    args = argparser.parse_args()

    try:
        setup_logger()
        main(args)
    except Exception as e:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
