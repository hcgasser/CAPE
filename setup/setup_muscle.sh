#!/bin/bash

folder=${PROGRAMS}/muscle
rm -rf $folder
mkdir $folder
cp ${SOFTWARE}/bioinf/muscle/muscle5.1.linux_intel64 $folder/

# create symbolic link
rm -f ${PROGRAMS}/bin/muscle
ln -s $folder/muscle5.1.linux_intel64 ${PROGRAMS}/bin/muscle
