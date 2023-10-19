from dotenv import load_dotenv
import os
import sys
from pathlib import Path

import argparse
import shlex
from tqdm.auto import tqdm
import yaml
import pdb
import logging

import torch
import torch.distributed as dist

import kit.globals as G
from kit.path import join
from kit.data import file_to_str, str_to_file, Split, DD
from kit.jobs import get_next_job_id
from kit.nn import move_list_to_device, move_dict_to_device
from kit.log import logger, setup_logger, log_info, log_warning, log_caller, caller
from kit.time import now
from kit.utils import get_class

tqdm.pandas()

def load_env(env, job_id, project_path):
    """ loads the environment from the yaml file

    :param env: id of the environment
    :param job_id (optional): id of the job, necessary for SLURM cluster, if NONE, the job_id is not set
    ;param project_path (optional): path to the project, overrides other paths
    """

    G.ENV = DD()
    G.PROJECT_ENV = DD()

    G.JOB['ID'] = None

    # load the paths
    if project_path is not None:
        G.ENV['PROJECT'] = project_path
    elif 'PF' in os.environ:
        G.ENV['PROJECT'] = os.environ['PF']
    else:
        G.ENV['PROJECT'] = os.getcwd()
    G.PROJECT_ENV['PROJECT'] = G.ENV.PROJECT

    print(f"ENV.PROJECT: '{G.ENV.PROJECT}'")
    yaml_env = file_to_str(os.path.join(G.ENV.PROJECT, "env", f"{env}.yaml"))
    yaml_env = yaml.load(yaml_env, Loader=yaml.FullLoader)


    PF = G.ENV.PROJECT
    PATHS = DD()

    G.ENV['ID'] = eval(yaml_env['ID'])
    G.ENV['KIT'] = eval(yaml_env['KIT'])
    G.JOB['ID'] = get_next_job_id(env) if job_id == "AUTO" else job_id
    program_subfolders = yaml_env['PROGRAM_SUBFOLDERS']

    for key, value in yaml_env['PATHS'].items():
        PATHS[key] = eval(value)
        if key in program_subfolders and G.PROGRAM is not None:
            G.PROJECT_ENV[key] = join(PATHS[key], G.PROJECT)
            G.ENV[key] = join(PATHS[key], G.PROGRAM)
        else:
            G.ENV[key] = G.PROJECT_ENV[key] = PATHS[key]

    G.ENV['JOBS'] = os.path.join(G.ENV.ARTEFACTS, 'jobs')
    G.ENV['JOB'] = join(G.ENV.JOBS, G.JOB.ID) if G.JOB.ID else None

    # load the rest of the environment specific configurations
    for sub in ["CONFIGS"]:
        G.ENV[sub] = DD()
        for key, value in yaml_env[sub].items():
            G.ENV[sub][key] = eval(value)


def load_task(task, **kwargs):
    """ loads the task from the yaml file

    :param task: task filename (without .yaml)
    :param kwargs: additional arguments to be passed to the DD.from_yaml function
    """

    G.TASK = DD()
    if task != "":
        task_file = os.path.join(G.ENV.CONFIG, "tasks", f"{task}.yaml")
        print(f"Load task file: {task_file}")
        G.TASK = DD.from_yaml(task_file, **kwargs)
        if "JOB" in G.TASK:
            G.TASK.PATHS = DD()
            G.TASK.PATHS.JOB = join(G.ENV.JOBS, G.TASK.JOB)
    else:
        log_warning(f"TASK NOT SPECIFIED")


def load_mhparams(mhparams, **kwargs):
    """ loads the model hyper-parameters (mhparams) from the yaml file

    :param mhparams: full mhparams filename including path
    :param kwargs: additional arguments to be passed to the DD.from_yaml function
    """

    if mhparams != "":
        G.MHPARAMS = G.MHPARAMS_standard.update_from_yaml(mhparams, **kwargs)
    else:
        log_warning(f"MHPARAMS NOT SPECIFIED")


def load_dhparams(dhparams, **kwargs):
    """ loads the data hyper-parameters (dhparams) from the yaml file

    :param dhparams: full dhparams filename including path
    :param kwargs: additional arguments to be passed to the DD.from_yaml function
    """

    if dhparams != "":
        G.DHPARAMS = DD.from_yaml(dhparams, **kwargs)
    else:
        log_warning(f"DHPARAMS NOT SPECIFIED")


def init(project, program, arg_string=None, logging_level=logging.INFO, project_path=None, create_job=True):
    """ initializes the supporting kit framework which has the following structure

    a 'project' is the name of the project
    a 'task' is the task that the project's cape_run.py has to achieve (defined by a yaml file)
    a 'job' is the concrete exection of the task. Accomplishing 'task' might require several 'jobs' (e.g. resume training a neural network)

    :param project: name of the project
    :param arg_string: string with arguments (project, env, task, job, debug)
    :param logging_level: logging level
    :param project_path: specify project path manually

    :return: args, args_unknown
    """

    global device 

    G.JOB = DD()
    G.MHPARAMS = DD()
    G.MHPARAMS_standard = DD() if G.MHPARAMS_standard is None else G.MHPARAMS_standard
    G.DHPARAMS = DD()

    if arg_string:
        G.ARGS, G.ARGS_UNKNOWN = parser.parse_known_args(shlex.split(arg_string))
    else:
        G.ARGS, G.ARGS_UNKNOWN = parser.parse_known_args()

    if G.ARGS.device == "cpu":
        device = "cpu"
    G.DEBUG = G.ARGS.debug
    if G.DEBUG:
        print("Started in DEBUG mode")
    print(f"Startup with args: {G.ARGS} ", end="")

    G.PROJECT = project
    G.PROGRAM = program
    G.DOMAIN = G.ARGS.domain

    # load environment
    load_env(G.ARGS.env, G.ARGS.job if create_job else None, project_path)

    # setup the python logging system
    log_file = None
    if G.MAIN_PROCESS:
        if create_job:
            log_file = join(G.ENV.JOBS, G.JOB.ID, f"job.log")
    else:
        logging_level = logging.ERROR
    setup_logger(logging_level, log_file)

    # load task definition
    load_task(G.ARGS.task)

    # Log summary to log files
    log_info(f"Start {G.PROJECT}-{G.TASK.ID}-{G.JOB.ID} on {G.ENV.ID} using {device}")

    log_info(f"G.ENV")
    for key, value in G.ENV.items():
        if key not in ["CONFIGS", "RUN"]:
            log_info(f"{key:<10s}... {value}")

    log_info(f"G.PROJECT_ENV")
    for key, value in G.PROJECT_ENV.items():
        if key not in ["CONFIGS", "RUN"]:
            log_info(f"{key:<10s}... {value}")

    log_info(f"Args: ")
    for arg, value in vars(G.ARGS).items():
        log_info(f"  {arg}={value} ")

    #
    # add logbook entry to task
    #
    if G.MAIN_PROCESS and G.TASK.ID is not None and G.JOB.ID is not None:
        str_to_file(f"{G.JOB.ID}\n", join(G.ENV.ARTEFACTS, "tasks", f"{G.TASK.ID}-{G.ENV.ID}.log"), append=True)

    return G.ARGS, G.ARGS_UNKNOWN


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# check if environment file is present
if os.path.exists(".env"):
    print("Load .env file")
    load_dotenv()

# add standard arguments
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="", help="task to be executed")
parser.add_argument('--job', type=str, default="AUTO")
parser.add_argument('--domain', type=str, default='d', help="the domain the code is running in (e.g. 'GFP')")
parser.add_argument('--env', type=str, default='py', help="py/notebook/mlp")
parser.add_argument('--device', type=str, default="cuda", help="cuda/cpu")
parser.add_argument('--debug', action=argparse.BooleanOptionalAction, help="should the program run in debug mode")

# the current model and trainer
model = None
trainer = None

