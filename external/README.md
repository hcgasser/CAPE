Please copy the 3rd party files into this folder on the host. 

From the below sources, obtain the necessary software and data as well as all required licences.
- netMHCpan: copy ``netMHCpan-4.1b.Linux.tar.gz`` and ``data.tar.gz`` from ``https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/`` into ``${CAPE}/external/bioinf/netMHCpan/``
- Rosetta: copy ``rosetta_bin_linux_3.13_bundle.tgz`` from ``https://www.rosettacommons.org/software/license-and-download`` into ``${CAPE}/external/bioinf/rosetta/``
- TMalign: copy ``TMalign.cpp`` and ``readme.c++.txt`` from ``https://zhanggroup.org/TM-align/`` into ``${CAPE}/external/bioinf/TMalign/``
- Muscle: copy ``muscle5.1.linux_intel64`` from ``https://drive5.com/muscle5/manual/install.html`` into ``${CAPE}/external/bioinf/muscle/``
- TransFun: clone repo on host
	- ``cd ${CAPE}/external``
	- ``git clone https://github.com/jianlin-cheng/TransFun.git``
	- ``cd TransFun``
	- ``curl https://calla.rnet.missouri.edu/rnaminer/transfun/data --output data.zip``
	- ``unzip data``


It should have the following structure:

```
external
├── bioinf
│   ├── muscle
│   │   └── muscle5.1.linux_intel64
│   ├── netMHCpan
│   │   ├── data.tar.gz
│   │   └── netMHCpan-4.1b.Linux.tar.gz
│   ├── rosetta
│   │   └── 3.13
│   │       └── rosetta_bin_linux_3.13_bundle.tgz
│   └── TMalign
│       ├── readme.c++.txt
│       └── TMalign.cpp
└── TransFun (subtree not shown)
```

To install the following programs in the container, please follow the below in the container (``make external``).

# Rosetta

After checking the licences run the following commands to install Rosetta (``. ${PF}/setup/setup_rosetta.sh``)
```
pushd ${PROGRAMS}/rosetta
tar -xzvf rosetta_bin_linux_3.13_bundle.tgz
rm rosetta_bin_linux_3.13_bundle.tgz
popd
```

# TransFun
The TransFun repo you downloaded can be found in the container under ``${PROGRAMS}/bioinf/TransFun``.
Please refer to the section "Installation" in ``${PROGRAMS}/bioinf/TransFun/README.md`` to install.

After checking the licences run the following commands to install TransFun (``. ${PF}/setup/setup_transfun.sh``)
```
pushd ${PROGRAMS}/TransFun/
curl https://calla.rnet.missouri.edu/rnaminer/transfun/data --output data.zip
unzip data
conda env create -f environment.yml
popd
```

# Muscle

After checking the licences run the following commands to install Muscle (``. ${PF}/setup/setup_muscle.sh``)

```
folder=${PROGRAMS}/muscle
rm -rf $folder
mkdir $folder
cp ${SOFTWARE}/bioinf/muscle/muscle5.1.linux_intel64 $folder/

# create symbolic link
rm -f ${PROGRAMS}/bin/muscle
ln -s $folder/muscle5.1.linux_intel64 ${PROGRAMS}/bin/muscle
```

# TMalign

After checking the licences run the following commands to install TMalign (``. ${PF}/setup/setup_TMalign.sh``)
```
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
```

# netMHCpan

After checking the licences run the following commands to install netMHCpan (``. ${PF}/setup/setup_netmhcpan.sh``)
```
pushd ${SOFTWARE}/bioinf/netMHCpan
rm -rf ${PROGRAMS}/netMHCpan
mkdir ${PROGRAMS}/netMHCpan
cp data.tar.gz ${PROGRAMS}/netMHCpan/
cp netMHCpan-4.1b.Linux.tar.gz ${PROGRAMS}/netMHCpan/

pushd ${PROGRAMS}/netMHCpan
tar -xzvf netMHCpan-4.1b.Linux.tar.gz
rm netMHCpan-4.1b.Linux.tar.gz
tar -xzvf data.tar.gz -C ./netMHCpan-4.1
rm data.tar.gz
pushd netMHCpan-4.1
progs=$(echo "$PROGRAMS" | sed 's/\//\\\//g')
replace="s/\/net\/sund-nas.win.dtu.dk\/storage\/services\/www\/packages\/netMHCpan\/4.1\/netMHCpan-4.1/${progs}\/netMHCpan\/netMHCpan-4.1/"
sed -i "$replace" netMHCpan
sed -i "s/TMPDIR  \/tmp/TMPDIR  \$NMHOME\/tmp/" netMHCpan
mkdir ./tmp
popd
popd
popd

#create symbolic link
rm -f ${PROGRAMS}/bin/netMHCpan
ln -s ${PROGRAMS}/netMHCpan/netMHCpan-4.1/netMHCpan ${PROGRAMS}/bin/netMHCpan
```


# CAPE-Packer

The following modifies Rosetta within the container (``. ${PF}/setup/setup_CAPE_Packer.sh``):


- ``source ${REPOS}/${REPO}/CAPE-Packer/to_rosetta/to_rosetta.sh``
- re-compile Rosetta within the container: 
	- ``cd $ROSETTA_PATH/main/source``
	- [Technical necessity since compiler needs this](https://rosettacommons.org/node/11709): 
		- in ``src/protocols/features/FeaturesReporter.fwd.hh`` add line ``#include <cstdint>`` after line ``#include <utility/pointer/owning_ptr.hh>``: 
                  ``sed -i "/#include <utility\/pointer\/owning_ptr.hh>/a #include <cstdint>" ./src/protocols/features/FeaturesReporter.fwd.hh``
		- ``scons -j8 mode=release bin``
