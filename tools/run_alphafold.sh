#!/usr/bin/env bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <cape repo path> <domain> <loch> [threshold]"
else
    cape=${1}
    domain=${2}
    loch=${3}


    . ${PF}/tools/pdb.sh


    threshold=200
    if [ $# -eq 4 ]; then
        threshold=${4}
    fi

    # create the pdbs representing the data
    echo "***************************************************"
    echo "**** Create support structures"
    echo "***************************************************"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE/${domain}.support.seq_hash ${loch} ${threshold}

    # create the XVAE pdbs
    echo "***************************************************"
    echo "**** Create CAPE-XVAE structures"
    echo "***************************************************"
    echo "**** CAPE-XVAE: baseline"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-XVAE/${domain}.CAPE-XVAE.baseline.final.seq_hash ${loch} ${threshold}
    echo "**** CAPE-XVAE: reduced visibility"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-XVAE/${domain}.CAPE-XVAE.reduce.final.seq_hash ${loch} ${threshold}
    echo "**** CAPE-XVAE: increased visibility"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-XVAE/${domain}.CAPE-XVAE.increase.final.seq_hash ${loch} ${threshold}
    echo "**** CAPE-XVAE: inc-nat visibility"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-XVAE/${domain}.CAPE-XVAE.inc-nat.final.seq_hash ${loch} ${threshold}

    # create the Packer pdbs
    echo "***************************************************"
    echo "**** Create CAPE-Packer structures"
    echo "***************************************************"
    echo "**** CAPE-Packer: baseline"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-Packer/${domain}.CAPE-Packer.baseline.final.seq_hash ${loch} ${threshold}
    echo "**** CAPE-Packer: reduced visibility"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-Packer/${domain}.CAPE-Packer.reduce.final.seq_hash ${loch} ${threshold}
    echo "**** CAPE-Packer: increased visibility"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-Packer/${domain}.CAPE-Packer.increase.final.seq_hash ${loch} ${threshold}
echo "**** CAPE-Packer: inc-nat visibility"
    pdb.AF.from_hashes ${cape}/artefacts/CAPE-Packer/${domain}.CAPE-Packer.inc-nat.final.seq_hash ${loch} ${threshold}

fi
