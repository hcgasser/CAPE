#!/bin/bash

pushd ${SOFTWARE}/bioinf/netMHCpan
rm -rf ${PROGRAMS}/netMHCpan
mkdir ${PROGRAMS}/netMHCpan

sudo apt-get install -y tcsh
tar -xzvf netMHCpan-4.1b.Linux.tar.gz -C ${PROGRAMS}/netMHCpan/
tar -xzvf data.tar.gz -C ${PROGRAMS}/netMHCpan/netMHCpan-4.1

pushd ${PROGRAMS}/netMHCpan/netMHCpan-4.1

progs=$(echo "$PROGRAMS" | sed 's/\//\\\//g')
replace="s/\/net\/sund-nas.win.dtu.dk\/storage\/services\/www\/packages\/netMHCpan\/4.1\/netMHCpan-4.1/${progs}\/netMHCpan\/netMHCpan-4.1/"
sed -i "$replace" netMHCpan
sed -i "s/TMPDIR  \/tmp/TMPDIR  \$NMHOME\/tmp/" netMHCpan
mkdir ./tmp

popd
popd

#create symbolic link
rm -f ${PROGRAMS}/bin/netMHCpan
ln -s ${PROGRAMS}/netMHCpan/netMHCpan-4.1/netMHCpan ${PROGRAMS}/bin/netMHCpan

# sed -i "s/\/net\/sund-nas.win.dtu.dk\/storage\/services\/www\/packages\/netMHCpan\/4.1\/netMHCpan-4.1/\/workspace\/progs\/netmhcpan\/netMHCpan-4.1/" netMHCpan
