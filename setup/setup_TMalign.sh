#!/usr/bin/env bash

folder=${PROGRAMS}/TMalign

rm -rf $folder
mkdir $folder
pushd ${SOFTWARE}/bioinf/TMalign
cp TMalign.cpp $folder/

pushd $folder
g++ -static -O3 -ffast-math -lm -o TMalign TMalign.cpp

popd
popd

rm -f ${PROGRAMS}/bin/TMalign
ln -s ${folder}/TMalign ${PROGRAMS}/bin/TMalign
