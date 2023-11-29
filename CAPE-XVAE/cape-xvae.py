#!/usr/bin/env python

""" Main script for CAPE-XVAE.

The tasks that are called with this script are for example training of CAPE-XVAE 
as well as sequence generation.

Additional arguments can be passed to the script. They are defined in the task.
In general, they will be in CAPITALS. In the YAML files, their values can be
    - values
    - "INPUT" (mandatory arguments) or
    - "OPTIONAL:Defaultvalue" (optional arguments).
"""


import importlib
import sys
import traceback

import kit
import kit.globals as G
from kit.bioinf import N_AA_FULL
from kit.data import DD
from kit.log import log_info


try:
    G.MHPARAMS_STANDARD = DD({"N_TOKENS": f"{N_AA_FULL}"})

    kit.init("CAPE", "CAPE-XVAE", docstring=__doc__)

    # log task information
    G.TASK.to_yaml(kit.path.join(G.ENV.JOB, "TASK.yaml"))

    log_info(f"load handler: {G.TASK.HANDLER}")
    module = importlib.import_module(f"CAPE.XVAE.handler.{G.TASK.HANDLER}")
    module.run()

except Exception as e:
    log_info("Exception: {e}")

    if G.ENV is None or "CONFIGS" not in G.ENV or G.ENV.CONFIGS.PM:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        kit.pdb.post_mortem(tb)
