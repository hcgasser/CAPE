#!/usr/bin/env python

import optuna
import importlib
import subprocess
import argparse
from datetime import datetime


def ls_studies(storage):
    study_summaries = optuna.study.get_all_study_summaries(storage=storage)
    text = f"{'Study Name':<50s} {'N-Trials':>10s} {'Best':>5s}\n"
    for study_summary in study_summaries:
        if study_summary.best_trial is not None:
            wert = f"{study_summary.best_trial.value:6.2f}"
            best_job = f"{study_summary.best_trial.user_attrs['JOB.ID']:>15s}"
        else:
            wert = f"{'---':6}"
            best_job = f"{'':>15s}"
        text += f"{study_summary.study_name:<50s} {study_summary.n_trials:>10} {wert} {best_job}\n"
    print(text)


def ls_trials(study, hparams=None, ascending=None):
    trials = [(trial.value, trial) for trial in study.trials]
    if ascending is not None:
        trials = sorted(trials, key=lambda x: x[0] if x[0] is not None else (float('inf') if ascending else float('-inf')))
    for value, trial in trials:
        print_trial(trial, hparams)


def print_trial(trial, hparams=None, show_date=False, show_intermediate=False):
    """ prints a trial in a nice format

    :param trial: the trial to be printed
    :param hparams: the hyperparameters to be printed (if None, all are printed)
    :param show_date: whether to show the start date of the trial
    :param show_intermediate: whether to show the number of intermediate values
    :return: None
    """

    params = ""

    datetime_start = trial.user_attrs['datetime_start'] if 'datetime_start' in trial.user_attrs else trial.datetime_start.timestamp()
    datetime_complete = trial.user_attrs['datetime_complete'] if 'datetime_complete' in trial.user_attrs else (
                        trial.datetime_complete.timestamp() if trial.datetime_complete is not None else None)
    if datetime_complete is not None:
        duration = (datetime.fromtimestamp(datetime_complete) - datetime.fromtimestamp(datetime_start)).total_seconds()
    else:
        duration = None

    for key, value in trial.params.items():
        if not hparams or (key in hparams):
            params += f"{key}: {f'{value:4}' if isinstance(value, int) else f'{value:8.2e}'} "

    text = (f"V: {trial.value:.3f} " if trial.value is not None else "V: ----- ")
    if show_intermediate:
        text += f"({len(trial.intermediate_values):2}) "
    text += f"{str(trial.state).split('.')[1]:8} "
    if show_date:
        text += f"{datetime.fromtimestamp(datetime_start).strftime('%Y-%m-%d %H:%M')}, "
    text += (f"{duration/3600:4.1f} h, " if duration is not None else "---- h, ")
    # text += f"{trial.number:3,} "
    text += f"{trial.number:<4} "
    text += f"{trial.user_attrs['JOB.ID']:<12s} " if 'JOB.ID' in trial.user_attrs else f"{'':<12s} "
    text += f"\n\t{params}"
    print(text)


def get_study(storage, study_name, direction="min"):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=f"{direction}imize"
    )
    setattr(study, "storage", storage)
    return study


def rm_study(study):
    optuna.delete_study(study_name=study.study_name, storage=study.storage)


def cp_trial(trial, params="NA", value="NA"):
    params = trial.params if params == "NA" else params
    value, state = (trial.value, trial.state) if value == "NA" else (value, optuna.trial._state.TrialState.COMPLETE)

    datetime_start = trial.user_attrs["datetime_start"] if "datetime_start" in trial.user_attrs else trial.datetime_start
    datetime_complete = trial.user_attrs["datetime_complete"] if "datetime_complete" in trial.user_attrs else trial.datetime_complete
    user_attrs = trial.user_attrs
    user_attrs['datetime_start'] = datetime_start.timestamp()
    if datetime_complete is not None:
        user_attrs['datetime_complete'] = datetime_complete.timestamp()
    else:
        user_attrs['datetime_complete'] = None

    return optuna.trial.create_trial(
        params=params,
        distributions=trial.distributions,
        value=value,
        intermediate_values=trial.intermediate_values,
        user_attrs=user_attrs,
        state=state
    )


def cp_study(study, study_name, adjust_trials=[]):
    old_trials = study.get_trials()
    keep_trials = []

    for trial in old_trials:
        if trial.number in adjust_trials:
            if adjust_trials[trial.number] != "del":
                keep_trials.append(cp_trial(trial, value=adjust_trials[trial.number]))
        else:
            keep_trials.append(cp_trial(trial))

    new_study = get_study(study.storage, study_name)
    rm_study(new_study)
    new_study = get_study(study.storage, study_name)
    new_study.add_trials(keep_trials)
    return new_study


def new_trial(x):
    global args, args_unknown

    x.set_user_attr("JOB.ID", args.job)

    tmp = args.call.split(".")
    module, function = tmp[:-1], tmp[-1]
    module = importlib.import_module(".".join(module))
    run_params = getattr(module, function)(args, x)
    print(f"Run: {run_params}")
    result = subprocess.run(run_params, capture_output=True)
    print("Finished")
    result = result.stdout.decode('UTF-8').split("\n")[-1]
    if result != "ERROR":
        result = float(result)

    print(f"Result: {result}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--call", type=str, default="",
                        help="python function to call, that takes as input the arguments and the trial and returns the parameters to subprocess.run as result")
    parser.add_argument("--storage", type=str, default=r"sqlite:///hyp.db",
                        help="the path of the hyperparameter database")
    parser.add_argument("--study", type=str, default="hyp", help="the name of the study")
    parser.add_argument("--direction", type=str, default="min", help="the direction to optimize the value to (min/max)")
    parser.add_argument("--job", type=str, default="", help="the job number")
    parser.add_argument("--env", type=str, default="py")
    args, args_unknown = parser.parse_known_args()

    study = get_study(args.storage, args.study, args.direction)

    study.optimize(new_trial, n_trials=1)