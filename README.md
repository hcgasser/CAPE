# Controlled Amount of Presented Epitopes (CAPE)

This is the code repository for the article: "Utility of language model and physics-based approaches in modifying MHC Class-I immune-visibility for the design of vaccines and therapeutics".

In this, two MHC Class I immune visiblity modification approaches were compared: CAPE-XVAE and CAPE-Rosetta.
This document explains how to install those and replicate the results.

## Installation

### General Requirements
- The programs in this repository require a Linux machine with docker and GPU support installed
- Alphafold needs to be installed (``https://github.com/google-deepmind/alphafold/``). Also, on the host system ``$ALPHAFOLD_REPO`` needs to point to the path of the AF repo and ``$ALPHAFOLD_DATA`` to the path of the database for the MSA. (CAPE will call ``python3 ${ALPHAFOLD_REPO}/docker/run_docker.py --data_dir=$ALPHAFOLD_DATA...``)

### Setup the container
If not indicated otherwise, commands should be run on the host system. Lines starting with 'H' need to be executed on the host, lines starting with 'C' in the container.

#### Clone the repository
- Clone repository to local machine: ``git clone git clone https://github.com/hcgasser/CAPE.git``
- Make environmental variable pointing to the repo folder: ``export CAPE=<path of repo folder>``

#### Create docker image
- Change into repo folder: ``cd $CAPE``
- Create docker image: ``${CAPE}/setup/create_image.sh`` (need to set a container password e.g. 'cape_pwd')

### Finish the setup of the container:
- Start docker image: ``docker run --name cape_container --gpus all -it -p 9000:9000 -v ${CAPE}:/CAPE cape``
- For licencing reasons we leave the installation of 3rd party software within the container to the user. Please refer to ``vim ${CAPE}/external/README.md`` for how to do this
- Install 'cape' conda environment: ``. setup/setup_conda.sh``
		
### Exit and restart the container
- C: the container can be exited every time with ``exit``
- H: the container can be restarted with ``docker start -i cape_container``
- (optional) H: to create an image out of the container ``docker commit cape_container cape_image``

## Run Experiments
In the below, we describe how to modify the immune visibility with the CAPE system for the example of the ``HIV-nef`` protein. If not stated differently, the below should be run in the container.
At first we set some standard values:
```
export DOMAIN="HIV_nef"
export MHC_Is="HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"
```
We include the ``. tools/set_HIV_nef.sh`` script to set these and the pretrained network

### Prepare data
- H: The input data needs to be present in the form of a single fasta file (e.g. ``HIV-1_nef.fasta``, Source: https://www.hiv.lanl.gov/components/sequence/HIV/search/search.html, Virus: HIV-1, Subtype: Any subtype, Genomic region: "Nef CDS" ~56k sequences, Download options: unaligned, amino acids). The file ``${CAPE}/configs/CAPE/dhparams/HIV_nef.yaml`` describes how the system should deal with this raw data. If this standard yaml is used, the fasta file ``HIV-1_nef.fasta`` needs to be placed in ``${CAPE}/data/raw/HIV-1_nef.fasta``
- Convert to model inputs: ``./cape.py --domain $DOMAIN --task data_raw_to_input``. This will ask for the 'DATA' name. This is the name of the yaml file described above, without path and extension within quotation marks (``"HIV_nef"``). The quotation marks are important, as the system evaluates the input. Alternatively, the command could also be run with an additional ``--DATA \"HIV_nef\"`` parameter. The concept of a 'domain' describes a set of common analysis. Several domains could deal with the same dataset - but for example use different models.
- Generate supports: ``./cape.py --domain $DOMAIN --task data_create_support --DATA \"HIV_nef\"``

### Prepare the MHC Class 1 position weight matrix predictor
Run the following in the container
- make output directory for the position weight matrices ``mkdir -p ${PF}/data/MHC_class_I``
- Generate the MHC Class 1 position weight matrices: ``MHC-I_rank_peptides.py --output ${PF}/data/MHC_class_I --alleles ${MHC_Is} --peptides_per_length 1000000``
- link the position weight matrices to CAPE: ``ln -s ${PF}/data/MHC_class_I/pwm ${PF}/data/input/pwm``


### CAPE-XVAE

#### Hyper-parameter search
- the following runs 10 hyper-parameter search experiment. Run as often as you wish. ``for ((i=1; i<=10; i++)); do cape-xvae.py --task hyp --domain ${DOMAIN} --MODEL \"XVAE_nef_32\" --DATA \"HIV_nef\"; done``. Some hyper-parameter combinations might 'crash' the program (e.g. too large for GPU). In this case just restart.
- you can list the results with the following command ``cape-xvae.py --domain $DOMAIN --task hyp_ls`` the last column of the first line of each entry states the job id (e.g. py_22). The job id of the pretrained HIV nef model is ``mlp_1606474`` and set as ``$CAPE_XVAE_JOB`` by ``set_HIV_nef.sh``.
- The job's subfolder can be found in ``/CAPE/artefacts/CAPE-XVAE/jobs/<job id>``

#### Plot training metrics
- Training metrics can be plotted using ``cape-xvae.py --task eval_plot_metrics --JOB \"$CAPE_XVAE_JOB\" --Y \"loss+loss_recon+loss_kl\"``. 
- The results can be found under ``figures`` in the job's subfolder

#### generate sequences
XVAE can generate clean (where impossible tokens and premature stop tokens are removed), as well as dirty sequences (where this is not the case). Here we also introduce the concept of the sequence hash file. To efficiently manage the multitude of sequences, structures, ... accross various analysis, the CAPE system generates a sequence hash for each sequence. 
If ``OUTPUT_FOLDER_PATH`` is ``None``, the generated sequences will be stored into ``/CAPE/artefacts/CAPE-XVAE/jobs/<job id>/generated/baseline`` (clean) and ``/CAPE/artefacts/CAPE-XVAE/jobs/<job id>/generated/dirty`` (dirty). 'clean' sequences will be referred to as *baseline* going forward. The dirty ones were actually only produced to check whether the system would actually regularly generate premature stop tokens and other impossible tokens.

Also, CAPE uses Loch, which is a directory and library, which stores all fasta, pdb, functional and molecular dynamics files associated with those sequences (hashes). *clean* sequences will also stored in this directory.
- C: ``cape-xvae.py --task generate --N_SEQS 100 --MODEL_ID \"$CAPE_XVAE_JOB\" --SEQ_HASH_FILE_PATH \"$PF/artefacts/CAPE-XVAE/${DOMAIN}.CAPE-XVAE.baseline.clean.seq_hash\" --CLEAN True --OUTPUT_FOLDER_PATH None``
- C: ``cape-xvae.py --task generate --N_SEQS 100 --MODEL_ID \"$CAPE_XVAE_JOB\" --SEQ_HASH_FILE_PATH \"$PF/artefacts/CAPE-XVAE/${DOMAIN}.CAPE-XVAE.baseline.dirty.seq_hash\" --CLEAN False --OUTPUT_FOLDER_PATH None``


#### modify immune visiblity:
The following commands randomly take a *natural* sequence (in the dataset used to train/validate/test the model) and run the deimmunization/immunization process on them. The results will be saved in the loch directory and the sequence hashes can be found in ``/CAPE/artefacts/CAPE-XVAE/${DOMAIN}.CAPE-XVAE.deimmunize.final.seq_hash``.
- C: deimmunize sequence: ``cape-xvae.py --domain $DOMAIN --task deimmunize --SEQ \"natural\" --MHCs \"$MHC_Is\" --MODEL_ID \"${CAPE_XVAE_JOB}:last\" --MODPARAMS \"modparams\"`` you can also run ``tools/CAPE-XVAE_deimmunize_100.sh`` to perform 100 deimmunizations
- C: immunize sequence: ``cape-xvae.py --domain $DOMAIN --task immunize --SEQ \"natural\" --MHCs \"$MHC_Is\" --MODEL_ID \"${CAPE_XVAE_JOB}:last\" --MODPARAMS \"modparams\"`` you can also run ``tools/CAPE-XVAE_immunize_100.sh`` to perform 100 immunizations


### CAPE-Packer

#### Start CAPE Packer Server
- C: Create new tmux window: ``tmux new-session -t CAPE_Packer_Server``
- C: ``cape_packer_server.py --port 12345 --pwm ${PF}/data/MHC_class_I/pwm``
- C: switch back to standard window: press ``Ctrl + b``then ``d`` 

#### Start CAPE-Packer baseline
- C: run CAPE Packer: ``cape_packer_from_seq_hashes.py --domain ${DOMAIN} --seq_hashes "${PF}/artefacts/CAPE/${DOMAIN}.support.seq_hash" --structure_path "${LOCH}/structures/AF/pdb" --structure_predictor AF --output_path "${PF}/artefacts/CAPE-Packer" --mhc_Is $MHC_Is --profile baseline`` and wait until finished

#### Start CAPE-Packer deimmunization
- C: run CAPE Packer: ``cape_packer_from_seq_hashes.py --domain ${DOMAIN} --seq_hashes "${PF}/artefacts/CAPE/${DOMAIN}.support.seq_hash" --structure_path "${LOCH}/structures/AF/pdb" --structure_predictor AF --output_path "${PF}/artefacts/CAPE-Packer" --mhc_Is $MHC_Is --profile deimmunize`` and wait until finished

#### Start CAPE-Packer immunization
- C: run CAPE Packer: ``cape_packer_from_seq_hashes.py --domain ${DOMAIN} --seq_hashes "${PF}/artefacts/CAPE/${DOMAIN}.support.seq_hash" --structure_path "${LOCH}/structures/AF/pdb" --structure_predictor AF --output_path "${PF}/artefacts/CAPE-Packer" --mhc_Is $MHC_Is --profile immunize`` and wait until finished



### Run CAPE-Eval

#### Start a Notebook Server
In the container:
- Create new tmux window: ``tmux new-session -t NB_Server``
- Change directory: ``cd ${PF}``
- Set password: ``jupyter server password``
- Start notebook server: ``jupyter server --port 9000``
- switch back to standard window: press ``Ctrl + b``then ``d`` 
- On the host open browser and go to ``http://localhost:9000/tree?`` and enter password defined above

#### Run evaluation notebook
On the host
- Convert the generated fast files to pdb files: ``${CAPE}/tools/run_alphafold.sh``
- Start the Notebook ``CAPE-Eval/cape-eval.ipynb``
