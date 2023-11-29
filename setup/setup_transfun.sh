#!/usr/bin/env bash

folder=${PROGRAMS}/TransFun
rm -rf $folder

cp -r ${PF}/external/TransFun ${PROGRAMS}

pushd $folder
conda env create -f environment.yml
# conda activate transfun
popd
