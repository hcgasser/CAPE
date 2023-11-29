#!/usr/bin/env python

""" Script to automatically run CAPE-Packer on a list of seq_hashes """


import os
import argparse
from datetime import datetime

import pandas as pd

from kit.log import log_info
from kit.path import join
from kit.data import str_to_file, file_to_str
from kit.loch import seq_hashes_to_file

from cape_packer import run


def main(_args):
    seq_hashes = [
        seq_hash
        for seq_hash in file_to_str(_args.seq_hashes).split("\n")
        if seq_hash != ""
    ]

    base_final_runtime_file_path = join(
        _args.output_path,
        f"{_args.domain}.CAPE-Packer.{_args.profile}.base_final_runtime.tsv",
    )
    if not os.path.exists(base_final_runtime_file_path):
        log_info(f"Creating {base_final_runtime_file_path}")
        str_to_file("base\tfinal\truntime\n", base_final_runtime_file_path, False)
    df_base_final_runtime = pd.read_csv(
        base_final_runtime_file_path, sep="\t"
    ).set_index("base")
    log_info(f"Using base_final_runtime_file_path: {base_final_runtime_file_path}")

    cape_packer_base_file = join(
        _args.output_path, f"{_args.domain}.CAPE-Packer.{_args.profile}.base.seq_hash"
    )
    cape_packer_final_file = join(
        _args.output_path, f"{_args.domain}.CAPE-Packer.{_args.profile}.final.seq_hash"
    )

    structure_predictor = _args.structure_predictor

    cnt = 1
    for seq_hash_base in seq_hashes:
        if not seq_hash_base in df_base_final_runtime.index:
            log_info(f"{cnt} of {len(seq_hashes)}, base: {seq_hash_base}")

            try:
                time_start = datetime.now()
                if _args.weight < 0:
                    raise Exception(f"weight must be not negative: {_args.weight}")

                seq_hash_final, generated_seq = run(
                    os.path.join(
                        _args.structure_path,
                        f"{seq_hash_base}_{structure_predictor}.pdb",
                    ),
                    join(_args.output_path, "generated", _args.profile),
                    _args.mhc_1_alleles,
                    _args.profile,
                    _args.weight,
                    True,
                    _args.port,
                    _args.rosetta_path,
                    _args.wd,
                )
                time_end = datetime.now()
                log_info(f"Generated: {seq_hash_final} {generated_seq}")

                runtime = int((time_end - time_start).total_seconds())

                seq_hashes_to_file([seq_hash_base], cape_packer_base_file, True)
                seq_hashes_to_file([seq_hash_final], cape_packer_final_file, True)
                str_to_file(
                    f"{seq_hash_base}\t{seq_hash_final}\t{runtime}\n",
                    base_final_runtime_file_path,
                    True,
                )
                df_base_final_runtime = pd.read_csv(
                    base_final_runtime_file_path, sep="\t"
                ).set_index("base")

            except Exception as e:
                log_info(e)

        cnt += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--domain", type=str, help="the prefix for the output files", required=True
    )
    argparser.add_argument(
        "--seq_hashes",
        type=str,
        help="the seq_hash file with the structures to optimize",
        required=True,
    )
    argparser.add_argument(
        "--structure_path",
        type=str,
        help="the path to the protein structure files",
        required=True,
    )
    argparser.add_argument(
        "--structure_predictor",
        type=str,
        help="the predicted structure to base the optimization on",
        required=True,
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        help="the path to output the seq_hash and base-final-runtime files to",
        required=True,
    )
    argparser.add_argument(
        "--mhc_1_alleles",
        type=str,
        help="+ separated MHC class 1 sequences "
        "(e.g. HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01)",
        required=True,
    )
    argparser.add_argument(
        "--profile",
        type=str,
        help="if no xml is supplied, the profile of the optimization",
        required=True,
    )
    argparser.add_argument(
        "--weight",
        type=float,
        default=1.0,
        help="the weight to put on the immuno score",
    )
    argparser.add_argument(
        "--port", type=int, help="the port of the server program", default=12345
    )
    argparser.add_argument(
        "--rosetta_path",
        type=str,
        help="path to the rosetta installation",
        required=True,
    )
    argparser.add_argument(
        "--wd",
        type=str,
        help="the working directory to run CAPE-Packer in "
        "(a temporary directory is used if not supplied)",
    )
    args = argparser.parse_args()

    main(args)
