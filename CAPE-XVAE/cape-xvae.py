#!/usr/bin/env python

import kit
import kit.globals as G
from kit.bioinf import n_aa
from kit.data import DD
from kit.log import log_info
import kit.globals as G

import os
import importlib
import sys, traceback

try:
    G.MHPARAMS_standard = DD({'N_TOKENS': f"{n_aa}"})
    
    kit.init('CAPE', 'CAPE-XVAE')

    # log task information
    G.TASK.to_yaml(kit.path.join(G.ENV.JOB, "TASK.yaml"))

    log_info(f"load handler: {G.TASK.HANDLER}")
    module = importlib.import_module(f"CAPE.XVAE.handler.{G.TASK.HANDLER}")
    module.run()

except Exception as e:
    log_info("Exception: {e}")
        
    if G.ENV is None or 'CONFIGS' not in G.ENV or G.ENV.CONFIGS.PM:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        kit.pdb.post_mortem(tb)
