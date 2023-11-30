<center><img src="logo.jpg" alt="Cape Logo" width="200" height="200"><br><br><br></center>


# Controlled Amplitude of Present Epitopes (CAPE)

This is the code repository for the article: **"Comparing a language model and a physics-based approach to modify MHC Class-I immune-visibility for the design of vaccines and therapeutics"**.

Protein therapeutics already have an arsenal of applications that include disrupting protein interactions, acting as potent vaccines, and replacing genetically deficient proteins. 
Therapeutics must avoid triggering unwanted immune-responses towards the therapeutic protein or viral vector proteins. 
In contrast, vaccines must support a robust immune-reaction targeting a broad range of pathogen variants. 
Therefore, computational methods modifying proteins' immunogenicity without disrupting function are needed. 
While many components of the immune-system can be involved in a reaction, we focus on CTL.
These target short peptides presented via the MHC-I pathway. 
To explore the limits of modifying the visibility of those peptides to CTL within the distribution of naturally occurring sequences, we developed a novel machine learning technique, **CAPE-XVAE**. 
It combines a language model with reinforcement learning to modify a protein's immune-visibility. 
Our results show that **CAPE-XVAE** effectively modifies the visibility of the HIV Nef protein to CTL. 
We contrast **CAPE-XVAE** to **CAPE-Packer**, a physics-based method we also developed.
Compared to **CAPE-Packer**, the machine learning approach suggests sequences that draw upon local sequence similarities in the training set. 
This is beneficial for vaccine development, where the synthetic sequence should be representative of the real viral population.
Additionally, the language model approach holds promise for preserving both known and unknown functional constraints, which are essential for the immune-modulation of therapeutic proteins.
In contrast, **CAPE-Packer**, emphasizes preserving the protein's overall fold and can reach greater extremes of immune-visibility, but falls short of capturing the sequence diversity of viral variants available to learn from.

Below we describe how to install the software used to obtain our results.

## Installation

### General Requirements
- The programs in this repository require a Linux machine with docker and GPU support installed
- Alphafold needs to be installed (``https://github.com/google-deepmind/alphafold/``).  
Also, on the host system ``$ALPHAFOLD_REPO`` needs to point to the path of the AF repo and ``$ALPHAFOLD_DATA`` to the path of the database for the MSA.  
(**CAPE** will call ``python3 ${ALPHAFOLD_REPO}/docker/run_docker.py --data_dir=$ALPHAFOLD_DATA...``)

### Setup the container
If not indicated otherwise, commands should be run on the host system. In particular, lines starting with 'H' need to be executed on the host, lines starting with 'C' in the container.

#### Clone the repository
- Clone repository to local machine: ``git clone git clone https://github.com/hcgasser/CAPE.git``
- Make environmental variable pointing to the repo folder: ``export CAPE=<path of repo folder>``

#### Create docker image
- Change into repo folder: ``cd $CAPE``
- Create docker image: ``make -B image`` (need to set a container password e.g. 'cape_pwd')

### Finish the setup of the container:
- Start docker image: ``docker run --name cape_container --gpus all -it -p 9000:9000 -v ${CAPE}:/CAPE cape``
- For licencing reasons we leave the installation of 3rd party software within the container to the user. Please refer to ``vim ${CAPE}/external/README.md`` for how to do this
- Install 'cape' conda environment: ``. ./setup/setup_conda.sh``
		
### Exit and restart the container
- C: the container can be exited every time with ``exit``
- H: the container can be restarted with ``docker start -i cape_container``
- (optional) H: to create an image out of the container ``docker commit cape_container cape_image``

## Run Experiments
In the below, we describe how to modify the immune visibility with the **CAPE** system for the example of the ``HIV-nef`` protein. If not stated differently, the below should be run in the container.
At first we set some standard values:
```
export DOMAIN="HIV_nef"
export MHC_Is="HLA-A*02:01+HLA-A*24:02+HLA-B*07:02+HLA-B*39:01+HLA-C*07:01+HLA-C*16:01"
```

### Prepare data
- H: The input data needs to be present in the form of a single fasta file  
(e.g. ``HIV-1_nef.fasta``, Source: https://www.hiv.lanl.gov/components/sequence/HIV/search/search.html, Virus: HIV-1, Subtype: Any subtype, Genomic region: "Nef CDS" ~56k sequences, Download options: unaligned, amino acids).  
The file ``${CAPE}/configs/CAPE/dhparams/HIV_nef.yaml`` describes how the system should deal with this raw data.  
If this yaml is used, the fasta file ``HIV-1_nef.fasta`` needs to be placed in ``${CAPE}/data/raw/HIV-1_nef.fasta``
- Convert to model inputs:  
``./cape.py --domain $DOMAIN --task data_raw_to_input``.  
This will ask for the 'DATA' name. This is the basename of the yaml file described above, without extension but importantly within quotation marks (``"HIV_nef"``).  
Alternatively, the command could also be run with an additional ``--DATA \"HIV_nef\"`` parameter.  
The concept of a 'domain' describes a set of common analysis. Several domains could deal with the same dataset - but for example use different models.
- Generate supports: ``./cape.py --domain $DOMAIN --task data_create_support --DATA \"HIV_nef\"``

### Prepare the MHC Class 1 position weight matrix predictor
Run the following in the container to generate the MHC Class 1 position weight matrices:  
``MHC-I_rank_peptides.py --output ${PF}/data/input/immuno/mhc_1/MhcPredictorPwm --alleles ${MHC_Is} --peptides_per_length 1000000``



### CAPE-XVAE

#### Hyper-parameter search
- The following command runs 10 hyper-parameter search experiment. Run as often as you wish. Some hyper-parameter combinations might 'crash' the program (e.g. too large for GPU). In this case just restart.  
``for ((i=1; i<=10; i++)); do cape-xvae.py --task hyp --domain ${DOMAIN} --MODEL \"XVAE_nef_32\" --DATA \"HIV_nef\"; done``
    - The ``--task`` parameter determines which yaml file specifies the task to perform (can be found in ``${PF}/configs/CAPE-XVAE/tasks/``)
    - The ``MODEL`` parameter determines which yaml file specifies the model hyper-parameters (can be found in ``${PF}/configs/CAPE-XVAE/mhparams/``). 
    - Finally, the ``DATA`` parameter determines the yaml file for the data (see "Prepare data")
- you can list the results with the following command ``cape-xvae.py --domain $DOMAIN --task hyp_ls`` the last column of the first line of each entry states the <job id> (e.g. py_22). The job's subfolder can be found in ``/CAPE/artefacts/CAPE-XVAE/jobs/<job id>``. Each job has a ``last`` checkpoint. So the corresponding checkpoint id would be ``<job id>:<ckpt>`` (e.g. ``py_12:last``). 
    - Set the checkpoint id of the best model to an environmental variable (e.g. ``export XVAE_CKPT_ID=py_12:last``). 
    - Also set the model id (e.g. ``export XVAE_MODEL_ID=py_12``)
The ckpt id of the pretrained HIV nef model is ``mlp_1606474:last`` and set as ``$XVAE_CKPT_ID`` by ``set_HIV_nef.sh``. Its model is is ``mlp_1606474`` and set as ``$XVAE_MODEL_ID``.

#### Plot training metrics
- Training metrics can be plotted by running  
``cape-xvae.py --task training_metrics --MODEL_ID \"$XVAE_MODEL_ID\" --Y \"loss+loss_recon+loss_kl\"``. 
- The results can be found under ``figures`` in the job's subfolder

#### generate sequences
**CAPE-XVAE** can generate clean (where impossible tokens and premature stop tokens are removed), as well as dirty sequences (where this is not the case). Here we also introduce the concept of the sequence hash file. To efficiently manage the multitude of sequences, structures, ... accross various analysis, the **CAPE** system generates a sequence hash for each sequence. 
If ``OUTPUT_FOLDER_PATH`` is not provided, the generated sequences will be stored into ``/CAPE/artefacts/CAPE-XVAE/jobs/<job id>/generated/baseline`` (clean) and ``/CAPE/artefacts/CAPE-XVAE/jobs/<job id>/generated/dirty`` (dirty). 'clean' sequences will be referred to as *baseline* going forward. The dirty ones were actually only produced to check whether the system would actually regularly generate premature stop tokens and other impossible tokens.

Also, **CAPE** uses Loch, which is a directory and library, which stores all fasta, pdb, functional and molecular dynamics files associated with those sequences (hashes). *clean* sequences will also be stored in this directory. To find sequences in Loch, we require their sequence hashes. These will be stored in the file specified with ``SEQ_HASH_FILE_PATH``. If this is not provided, the standard file path will be used (``/CAPE/artefacts/CAPE-XVAE/${DOMAIN}.CAPE-XVAE.baseline.clean/dirty.seq_hash``). In the container run:

- to generate 100 clean sequences:  
``cape-xvae.py --task generate --domain $DOMAIN \``  
``--CLEAN True --N_SEQS 100 --CKPT_ID \"$XVAE_CKPT_ID\"``
- to generate 100 dirty sequences:  
``cape-xvae.py --task generate --domain $DOMAIN \``  
``--CLEAN False --N_SEQS 100 --CKPT_ID \"$XVAE_CKPT_ID\" ``


#### modify immune visiblity:
The following commands (run in the container) randomly take a *natural* sequence (in the dataset used to train/validate/test the model) and run the immune-visibility modification process on them. The results will be saved in the loch directory and the sequence hashes can be found in ``/CAPE/artefacts/CAPE-XVAE/${DOMAIN}.CAPE-XVAE.${PROFILE}.final.seq_hash``.

- Repeat the following for each immune-visibility profile (``reduced/increased/inc-nat``)  
``export PROFILE=reduced``
- design sequences for the ``$PROFILE`` immune-visible profile: 
    - ``cape-xvae.py --task modify --domain $DOMAIN \``  
      ``--PROFILE \"${PROFILE}\" --MODPARAMS \"modparams\" --MHCs \"$MHC_Is\" \``  
      ``--SEQ \"natural\" --SEQ_ENC_STOCHASTIC True  \``  
      ``--CKPT_ID \"${XVAE_CKPT_ID}\"``  
    - you can also run ``tools/run_CAPE-XVAE_modify.sh ${PROFILE} 100`` to perform 100 designs


### CAPE-Packer

#### Start CAPE Packer Server
- C: Create new tmux window: ``tmux new-session -t CAPE_Packer_Server``
- C: ``cape_packer_server.py --port 12345 --pwm_path ${PF}/data/input/immuno/mhc_1/MhcPredictorPwm/pwm --input_files_path ${PF}/data/input/HIV_nef_full``
- C: switch back to standard window: press ``Ctrl + b`` then ``d`` 

#### Run CAPE-Packer client
Before the next steps, the PDB files need to be present in the ``structure_path``. 
Next to other methods, this can be achieve via running the following on the host system:  
H: ``./tools/run_alphafold.sh ${CAPE} HIV_nef ${CAPE}/artefacts/CAPE/loch``

- Repeat the following for each immune-visibility profile (``baseline/reduced/increased/inc-nat``)  
``export PROFILE=reduced``
- design sequences for the ``$PROFILE`` immune-visible profile:  
``cape_packer_from_seq_hashes.py --domain ${DOMAIN} \``
``--profile ${PROFILE} --mhc_1_alleles $MHC_Is \``
``--structure_path "${LOCH}/structures/AF/pdb" --structure_predictor AF \``
``--seq_hashes "${PF}/artefacts/CAPE/${DOMAIN}.support.seq_hash" \``
``--output_path "${PF}/artefacts/CAPE-Packer" \``
``--rosetta_path $ROSETTA_PATH``
    - ``--structure_path``: the path where the program can find the pdb files
    - ``--structure_predictor``: the name of the used structure predictor (the 3D structure filename is then ``<seq hash from seq_hash file>_<structure predictor>.pdb``)
    - ``--seq_hashes``: the file containing the sequence hashes of the 3D structures
    - ``--output_path``: where to output the generated sequences to
    - ``--rosetta_path``: path to the rosetta system


### Run CAPE-Eval

#### Start a Notebook Server
In the container:
- Create new tmux window: ``tmux new-session -t NB_Server``
- Change directory: ``cd ${PF}``
- Set password: ``jupyter server password``
- Start notebook server: ``jupyter server --port 9000``
- switch back to standard window: press ``Ctrl + b``then ``d`` 

#### Produce 3D Structures, MD Simulations, Functional Predicitons
- H: To produce the 3D structures run ``${CAPE}/tools/run_alphafold.sh``
- C: To predict the GO Molecular Functions run ``${PF}/tools/run_transfun.sh``
- C: This will take a long time. So we suggest you are also only running it for a sample. For each of the seq-hash files 
    - ``${PF}/artefacts/CAPE/HIV_nef.support.seq_hash``
    - ``${PF}/artefacts/CAPE-XVAE/HIV_nef.CAPE-XVAE.baseline.final.seq_hash``
    - ``${PF}/artefacts/CAPE-XVAE/HIV_nef.CAPE-XVAE.reduced.final.seq_hash``
    - ``${PF}/artefacts/CAPE-XVAE/HIV_nef.CAPE-XVAE.increased.final.seq_hash``
    - ``${PF}/artefacts/CAPE-XVAE/HIV_nef.CAPE-XVAE.inc-nat.final.seq_hash``
    - ``${PF}/artefacts/CAPE-Packer/HIV_nef.CAPE-Packer.baseline.final.seq_hash``
    - ``${PF}/artefacts/CAPE-Packer/HIV_nef.CAPE-Packer.reduced.final.seq_hash``
    - ``${PF}/artefacts/CAPE-Packer/HIV_nef.CAPE-Packer.increased.final.seq_hash``
    - ``${PF}/artefacts/CAPE-Packer/HIV_nef.CAPE-Packer.inc-nat.final.seq_hash``

    run

```
export SEQ_HASH_FILE=<insert path to seq_hash file>
tools/run_for_seq_hashes.sh "${PF}/tools/MD/md.py --pdb ${LOCH}/structures/AF/pdb/#SEQ_HASH#_AF.pdb --output ${LOCH}/dynamics" $SEQ_HASH_FILE
```

#### Run evaluation notebook
- On the host open browser and go to ``http://localhost:9000/tree?`` and enter password defined above
- Open and run the notebook in ``CAPE-Eval/cape-eval.ipynb``
