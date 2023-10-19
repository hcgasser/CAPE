#!/bin/bash

TO_ROSETTA=${PF}/CAPE-Packer/to_rosetta

MHC_EPITOPE_ENERGY_PATH=${ROSETTA_PATH}/main/source/src/core/scoring/mhc_epitope_energy
cp -f ${TO_ROSETTA}/MHCEpitopePredictorClient.hh ${MHC_EPITOPE_ENERGY_PATH}/
cp -f ${TO_ROSETTA}/MHCEpitopePredictorClient.fwd.hh ${MHC_EPITOPE_ENERGY_PATH}/
cp -f ${TO_ROSETTA}/MHCEpitopePredictorClient.cc ${MHC_EPITOPE_ENERGY_PATH}/
cp -f ${TO_ROSETTA}/MHCEpitopeEnergySetup.cc ${MHC_EPITOPE_ENERGY_PATH}/

cp -f ${TO_ROSETTA}/core.3.src.settings ${ROSETTA_PATH}/main/source/src/

cp -f ${TO_ROSETTA}/Client.mhc ${ROSETTA_PATH}/main/database/scoring/score_functions/mhc_epitope/
