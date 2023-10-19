#!/bin/bash

source ${PF}/CAPE-Packer/to_rosetta/to_rosetta.sh
pushd $ROSETTA_PATH/main/source
sed -i "/#include <utility\/pointer\/owning_ptr.hh>/a #include <cstdint>" ./src/protocols/features/FeaturesReporter.fwd.hh
scons -j8 mode=release bin
popd
