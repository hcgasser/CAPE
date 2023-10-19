#!/usr/bin/env bash

conda activate base
conda env remove --name cape
conda create -y -n cape python=3.10

conda activate cape
echo "conda activate cape" >> ${HOME}/.bashrc

pip install --upgrade pip
pip install -r ${PF}/setup/requirements.txt

conda install -c conda-forge -y openmm
conda install -c conda-forge -y libstdcxx-ng
conda install -c conda-forge -y mdtraj

jupyter server --generate-config
echo "c.ServerApp.ip = '0.0.0.0'" >> ${HOME}/.jupyter/jupyter_notebook_config.py
