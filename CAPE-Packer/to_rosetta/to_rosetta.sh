#!/bin/bash

TO_ROSETTA=${PF}/CAPE-Packer/to_rosetta
MHC_EPITOPE_ENERGY_PATH=${ROSETTA_PATH}/main/source/src/core/scoring/mhc_epitope_energy

cp -f ${TO_ROSETTA}/Client.mhc ${ROSETTA_PATH}/main/database/scoring/score_functions/mhc_epitope/

# modify core.3.src.settings
sh ${TO_ROSETTA}/core.3.src.settings.sh ${ROSETTA_PATH}/main/source/src/core.3.src.settings

# modify MHCEpitopeEnergySetup.cc
sh ${TO_ROSETTA}/MHCEpitopeEnergySetup.cc.sh $MHC_EPITOPE_ENERGY_PATH/MHCEpitopeEnergySetup.cc

# create MHCEpitopePredictorClient.hh
cp -f $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorMatrix.hh $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorClient.hh
sh ${TO_ROSETTA}/MHCEpitopePredictorClient.hh.sh $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorClient.hh

# create MHCEpitopePredictorClient.fwd.hh
cp -f $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorMatrix.fwd.hh $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorClient.fwd.hh
sh ${TO_ROSETTA}/MHCEpitopePredictorClient.fwd.hh.sh $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorClient.fwd.hh

# create MHCEpitopePredictorClient.cc
cp -f $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorMatrix.cc $MHC_EPITOPE_ENERGY_PATH/MHCEpitopePredictorClient.cc
sh ${TO_ROSETTA}/MHCEpitopePredictorClient.cc.sh $MODIFIED/MHCEpitopePredictorClient.cc


