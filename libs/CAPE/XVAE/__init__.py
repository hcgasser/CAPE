import os

import kit
import kit.globals as G
from kit.path import join
from kit.globals import ENV
from kit.jobs import get_job_info


def decompose_ckpt_id(ckpt_id):
    ckpt_id = ckpt_id.split(":")
    if len(ckpt_id) == 2:
        model_id, ckpt_name = ckpt_id  # job_id, ckpt
    else:
        raise ValueError("Invalid ckpt_id")
    return model_id, ckpt_name


def get_model_dir_path(model_id, jobs_dir_path=None, check_exists=True):
    jobs_dir_path = jobs_dir_path if jobs_dir_path is not None else G.ENV.JOBS
    model_dir_path = os.path.join(jobs_dir_path, model_id)
    if check_exists and not os.path.exists(model_dir_path):
        raise FileNotFoundError(f"Model '{model_id}' not found.")
    return model_dir_path


def get_ckpt_dir_path(ckpt_id, jobs_dir_path=None, check_exists=True):
    model_id, ckpt_name = decompose_ckpt_id(ckpt_id)
    model_dir_path = get_model_dir_path(model_id, jobs_dir_path, check_exists)

    ckpt_dir_path = os.path.join(model_dir_path, ckpt_id)
    ckpt_path = os.path.join(model_dir_path, "checkpoints", ckpt_name + ".ckpt")
    if check_exists and not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_id}' not found.")
    return model_id, ckpt_name, ckpt_dir_path, ckpt_path


def add_shortcut_to_artefacts(sub_dir, entity_dir_path, entity_id):
    if not os.path.islink(os.path.join(G.ENV.ARTEFACTS, sub_dir, entity_id)):
        tgt = os.path.join(entity_dir_path, sub_dir)
        src = join(G.ENV.ARTEFACTS, sub_dir)
        os.symlink(os.path.relpath(tgt, src), os.path.join(src, entity_id))


def load_ckpt(ckpt_id, jobs_directory_path=None):
    """Loads a model from a checkpoint into the global kit.MODEL variable.

    Args:
        ckpt_id (str): The checkpoint id of the checkpoint to load.
            The ckpt id is of the form <model_id>:<ckpt_name>
            where <ckpt_name> is optional (then 'last').
            The model id is the job id of the first job used to
            train the model
        jobs_directory_path (str, optional): The path to the jobs directory. Defaults to None.

    Returns:
        str: The path to the job directory.
    """

    model_id, ckpt_name = decompose_ckpt_id(ckpt_id)

    jobs_directory_path = (
        jobs_directory_path if jobs_directory_path is not None else G.ENV.JOBS
    )
    job_info = get_job_info(
        model_id,
        mhparams=True,
        dhparams=True,
        ckpt=ckpt_name,
        jobs_directory_path=jobs_directory_path,
    )
    G.MHPARAMS, G.DHPARAMS, (_, ckpt_path) = (
        job_info[k] for k in ["mhparams", "dhparams", "ckpt"]
    )

    kit.MODEL = G.MHPARAMS.CLS.load_from_checkpoint(
        ckpt_path
    )  # load model from checkpoint

    # necessary to allow gradients to flow back to the latent variable
    for block in kit.MODEL.decoder.decoder.blocks:
        for head in block.xa.heads:
            head.recalc_masked = True

    kit.MODEL.to(kit.DEVICE)
    kit.MODEL.eval()

    return join(jobs_directory_path, model_id, ckpt_id)
