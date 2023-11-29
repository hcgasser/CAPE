#!/usr/bin/env python

""" Main script for CAPE project wide operations

Data preprocessing is an example of a project wide operation - required for many 
CAPE methods (e.g. CAPE-XVAE, CAPE-Packer, ...).

Usaga:
    python cape.py --task <task.yaml> --env <env.yaml> --domain <domain_name> ...

    --task: task configuration file without extension. It is in YAML format and can be
        found in configs/CAPE-XVAE/tasks/
        Most importantly, it specifies the handler to be used for the task.
        The handler is a python file that can be found in libs/CAPE/XVAE/handler/

    --env: environment configuration file without extension. It is in YAML format and
        can be found in env/. These yaml files specify the directory structure which
        can differ dependent on the envirionment the program runs in.
        (e.g. local, cluster, etc.)

    --domain: allows to run independent experiments in parallel. 
        It is a unique string to distinguish them

    ...: additional arguments that are defined in the task configuration file.
        In general, they will be in CAPITALS. Their values in the YAML can be
        - values
        - "INPUT" (mandatory arguments) or 
        - "OPTIONAL:Defaultvalue" (optional arguments).

"""

import importlib
import sys
import traceback

import kit
import kit.globals as G
from kit.log import log_info

try:
    kit.init("CAPE", "CAPE", create_job=False)

    log_info(f"load handler: {G.TASK.HANDLER}")
    module = importlib.import_module(f"CAPE.handler.{G.TASK.HANDLER}")
    module.run()

except Exception as e:
    log_info("Exception: {e}")

    if G.ENV is None or "CONFIGS" not in G.ENV or G.ENV.CONFIGS.PM:
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        kit.pdb.post_mortem(tb)
