import numpy as np
import torch

import kit
import kit.globals as G
from kit.path import join
from kit.globals import ENV
from kit.jobs import get_job_info


def load_model(model_id):
    model_id = model_id.split(":")
    if len(model_id) == 2:
        job_id, ckpt = model_id
    elif len(model_id) == 1:
        job_id, ckpt = model_id[0], "last"
    else:
        raise ValueError("Invalid model_id")

    job_info = get_job_info(job_id, mhparams=True, dhparams=True, ckpt="last")
    G.MHPARAMS, G.DHPARAMS, (ckpt_job_id, ckpt_path) = (job_info[k] for k in ['mhparams', 'dhparams', 'ckpt'])

    kit.model = G.MHPARAMS.CLS.load_from_checkpoint(ckpt_path)  # load model from checkpoint

    # necessary to allow gradients to flow back to the latent variable
    for block in kit.model.decoder.decoder.blocks:
        for head in block.xa.heads:
            head.recalc_masked = True

    kit.model.to(kit.device)
    kit.model.eval()

    return join(G.ENV.JOBS, job_id)