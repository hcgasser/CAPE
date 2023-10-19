#!/usr/bin/env python

import os
import sys
import subprocess
import sys, traceback
import argparse
import pdb
import tempfile

from kit.path import join
from kit.log import log_info
from kit.data import str_to_file, remove_lines_between
from kit.loch.path import get_fasta_file_path
from kit.loch.utils import get_seq_hash
from kit.bioinf.pdb import pdb_to_seqs
from kit.bioinf.fasta import seqs_to_fasta


def run_CAPE_Packer(input, output, mhc_Is, profile, weight, to_loch, wd_path=None):
    if wd_path is None:
        with tempfile.TemporaryDirectory() as tmp_wd_path:
            return _run_CAPE_Packer(input, output, mhc_Is, profile, weight, to_loch, tmp_wd_path)
    return _run_CAPE_Packer(input, output, mhc_Is, profile, weight, wd_path)
        

def _run_CAPE_Packer(input, output, mhc_Is, profile, weight, to_loch, wd_path):
    cwd = os.getcwd()
    os.chdir(wd_path)
    seq_hash = None
    log_info(f"Run CAPE-Packer in {wd_path}")


    mhc_file_path = os.path.join(os.environ['ROSETTA_PATH'], 'main', 'database', 'scoring', 'score_functions',
                                 'mhc_epitope', 'Client.mhc')
    str_to_file(get_mhc_file(mhc_Is), mhc_file_path)
    log_info(f"    in {mhc_file_path}")

    xml_file_path = os.path.join(wd_path, 'script.xml')
    str_to_file(get_xml_file(profile, weight=weight, mhc_file_path=mhc_file_path), xml_file_path)

    log_info(f"    for {profile} using {weight} in {wd_path}")

    # run rosetta packer
    command = ['rosetta_scripts.default.linuxgccrelease',
                '-s', input,
                '-parser:protocol', xml_file_path,
                '-nstruct', '1']
    result = subprocess.run(command, capture_output=True)

    os.chdir(cwd)
    # process the result
    pdb_file = os.path.join(wd_path, f"{os.path.splitext(os.path.basename(input))[0]}_0001.pdb")
    if os.path.exists(pdb_file):
        remove_lines_between(pdb_file, '^#BEGIN_POSE_ENERGIES_TABLE.*', '^#END_POSE_ENERGIES_TABLE.*')
        remove_lines_between(pdb_file, '^TER.+')
        CAPE_Packer_seq = pdb_to_seqs(pdb_file)[0]['A']
        seq_hash = get_seq_hash(CAPE_Packer_seq)
        seqs_to_fasta([CAPE_Packer_seq], join(output, f"{seq_hash}.fasta"), [seq_hash])
        if to_loch:
            seqs_to_fasta([CAPE_Packer_seq], get_fasta_file_path(seq_hash), [seq_hash])
        return seq_hash, CAPE_Packer_seq
    else:
        raise Exception(f"{pdb_file} does not exist: {result.stderr} {result.stdout}")


def get_mhc_file(mhc_Is):
    mhc_Is = mhc_Is.split('+')
    s_alleles = ",".join(mhc_Is) # HLA-A*02:01,HLA-A*24:02,HLA-B*07:02,HLA-B*39:01,HLA-C*07:01,HLA-C*16:01
    mhc_file = f"""
method Client 12345
alleles {s_alleles}
threshold 5
"""
    log_info(f"Generated MHC file for {s_alleles}")
    return mhc_file


def get_xml_file(profile, weight, mhc_file_path):
    if profile == 'baseline':
        weight = 0.0
    elif profile == 'immunize':
        weight = -weight
    elif profile == 'deimmunize':
        pass
    else:
        raise Exception(f"Unknown profile: {profile}")

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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--input', type=str, help="the pdb filename to base the optimization on", required=True)
    argparser.add_argument('--output', type=str, help="the output folder for the fasta file", required=True)
    argparser.add_argument('--mhc_Is', type=str, help="+ separated MHC class 1 sequences (e.g. HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01)", required=True)
    argparser.add_argument('--profile', type=str, help="if no xml is supplied, the profile of the optimization", required=True)
    argparser.add_argument('--weight', type=float, help="the weight to put on the immuno score")
    argparser.add_argument('--to_loch', type=str, help="should the results be stored in the loch. True/False", default='True')
    args = argparser.parse_args()

    try:
        to_loch = True if args.to_loch == 'True' else False
        seq_hash, CAPE_Packer_seq = run_CAPE_Packer(args.input, args.output, args.mhc_Is, args.profile, args.weight, to_loch)
        log_info(f"Generated: {seq_hash} {CAPE_Packer_seq}")
    except Exception as e:
        log_info(e)
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
