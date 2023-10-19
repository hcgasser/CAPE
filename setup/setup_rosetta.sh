#!/bin/bash

pushd ${SOFTWARE}/bioinf/rosetta
rm -rf ${PROGRAMS}/rosetta
mkdir ${PROGRAMS}/rosetta

sudo apt-get install -y libjson-c-dev scons libz-dev
tar -xzvf ./3.13/rosetta_bin_linux_3.13_bundle.tgz -C ${PROGRAMS}/rosetta

pushd ${PROGRAMS}/rosetta
export ROSETTA_PATH="${PROGRAMS}/rosetta/rosetta_bin_linux_2021.16.61629_bundle"
echo 'ROSETTA_PATH="${PROGRAMS}/rosetta/rosetta_bin_linux_2021.16.61629_bundle"' >> ${HOME}/.bashrc
echo "PATH=${PATH}:${ROSETTA_PATH}/main/source/bin" >> ${HOME}/.bashrc

popd
popd
