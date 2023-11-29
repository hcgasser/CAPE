#!/usr/bin/env python

import os
import sys
import subprocess
import traceback
import argparse
import pdb
from kit.utils import temp_working_directory

from kit.path import join
from kit.log import log_info
from kit.data import str_to_file
from kit.data.utils import remove_lines_between
from kit.loch.path import get_fasta_file_path
from kit.loch.utils import get_seq_hash
from kit.bioinf.pdb import pdb_to_seqs
from kit.bioinf.fasta import seqs_to_fasta

from CAPE.profiles import Profile


def run(
    input_file_path,
    output_dir_path,
    mhc_1_alleles,
    profile,
    weight,
    to_loch,
    port,
    rosetta_path,
    wd_path=None,
):
    profile = Profile(profile)
    with temp_working_directory(wd_path) as tmp_wd_path:
        return _run(
            input_file_path,
            output_dir_path,
            mhc_1_alleles,
            profile,
            weight,
            to_loch,
            port,
            rosetta_path,
            tmp_wd_path,
        )


def _run(
    input_file_path,
    output_dir_path,
    mhc_1_alleles,
    profile,
    weight,
    to_loch,
    port,
    rosetta_path,
    wd_path,
):
    log_info(wd_path)
    cwd = os.getcwd()
    os.chdir(wd_path)
    seq_hash = None
    log_info(f"Run CAPE-Packer in {wd_path}")

    mhc_file_path = os.path.join(
        rosetta_path,
        "main",
        "database",
        "scoring",
        "score_functions",
        "mhc_epitope",
        "Client.mhc",
    )
    str_to_file(get_mhc_file(mhc_1_alleles, port, profile), mhc_file_path)
    log_info(f"    with {mhc_file_path}")

    xml_file_path = os.path.join(wd_path, "script.xml")
    str_to_file(
        get_xml_file(profile, weight=weight, mhc_file_path=mhc_file_path), xml_file_path
    )

    log_info(f"    for {profile} using {weight} in {wd_path}")

    # run rosetta packer
    command = [
        "rosetta_scripts.default.linuxgccrelease",
        "-s",
        input_file_path,
        "-parser:protocol",
        xml_file_path,
        "-nstruct",
        "1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)

    os.chdir(cwd)
    # process the result
    pdb_file = os.path.join(
        wd_path, f"{os.path.splitext(os.path.basename(input_file_path))[0]}_0001.pdb"
    )
    if os.path.exists(pdb_file):
        remove_lines_between(
            pdb_file, "^#BEGIN_POSE_ENERGIES_TABLE.*", "^#END_POSE_ENERGIES_TABLE.*"
        )
        remove_lines_between(pdb_file, "^TER.+")
        generated_seq = pdb_to_seqs(pdb_file)[0]["A"]
        seq_hash = get_seq_hash(generated_seq)
        seqs_to_fasta(
            [generated_seq], join(output_dir_path, f"{seq_hash}.fasta"), [seq_hash]
        )
        if to_loch:
            seqs_to_fasta([generated_seq], get_fasta_file_path(seq_hash), [seq_hash])
        return seq_hash, generated_seq

    raise Exception(f"{pdb_file} does not exist: {result.stderr} {result.stdout}")


def get_mhc_file(mhc_1_alleles, port, profile):
    mhc_1_alleles = mhc_1_alleles.split("+")
    s_alleles = ",".join(
        mhc_1_alleles
    )  # HLA-A*02:01,HLA-A*24:02,HLA-B*07:02,HLA-B*39:01,HLA-C*07:01,HLA-C*16:01

    reward_visible_artificial = profile.reward_visible_artificial()
    reward_visible_natural = profile.reward_visible_natural()

    mhc_file = f"""
method Client {port}
alleles {s_alleles}
rewards {reward_visible_artificial} {reward_visible_natural}
"""
    log_info(f"Generated MHC file for {s_alleles}")
    return mhc_file


def get_xml_file(profile, weight, mhc_file_path):
    if not isinstance(profile, Profile):
        raise Exception(f"Unknown profile: {profile}")

    if profile == Profile.BASE:
        weight = 0.0
    else:
        weight = -weight

    xml_file = f"""
<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="ref_deimm" weights="ref2015.wts" >
            <Reweight scoretype="mhc_epitope" weight="{weight}" />
            <Set mhc_epitope_setup_file="{os.path.basename(mhc_file_path)}" />
        </ScoreFunction>
    </SCOREFXNS>
    <RESIDUE_SELECTORS>
    </RESIDUE_SELECTORS>
    <PACKER_PALETTES>
    </PACKER_PALETTES>
    <TASKOPERATIONS>
    </TASKOPERATIONS>
    <MOVE_MAP_FACTORIES>
    </MOVE_MAP_FACTORIES>
    <SIMPLE_METRICS>
    </SIMPLE_METRICS>
    <FILTERS>
    </FILTERS>
    <MOVERS>
    <PackRotamersMover name="pack" scorefxn="ref_deimm" />
    </MOVERS>
    <PROTOCOLS>
    <Add mover="pack" />
    </PROTOCOLS>
    <OUTPUT scorefxn="ref_deimm" />
</ROSETTASCRIPTS>"""

    log_info(f"Generated XML script for {profile} using {weight}")
    return xml_file


def main(_args):
    try:
        seq_hash, generated_seq = run(
            _args.input,
            _args.output,
            _args.mhc_1_alleles,
            _args.profile,
            _args.weight,
            _args.to_loch == "True",
            _args.port,
            _args.rosetta_path,
        )
        log_info(f"Generated: {seq_hash} {generated_seq}")
    except Exception as e:
        log_info(e)
        _, _, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "--input",
        type=str,
        help="the pdb filename to base the optimization on",
        required=True,
    )
    argparser.add_argument(
        "--output", type=str, help="the output folder for the fasta file", required=True
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
        "--weight", type=float, help="the weight to put on the immuno score"
    )
    argparser.add_argument(
        "--to_loch",
        type=str,
        help="should the results be stored in the loch. True/False",
        default="True",
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
    args = argparser.parse_args()

    main(args)
