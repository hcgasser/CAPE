import kit.globals as G
from kit.hyp import get_study, ls_trials
import CAPE.XVAE.handler.train


def run():
    study = get_study(G.TASK.STORAGE, G.DOMAIN, G.TASK.DIRECTION[:3])

    if G.TASK.ID == "hyp":

        def new_trial(x):
            x.set_user_attr("JOB.ID", G.JOB.ID)
            return CAPE.XVAE.handler.train.fit(trial=x)

        if G.MAIN_PROCESS:
            study.optimize(new_trial, n_trials=1)
        else:
            trial = study.trials[-1]
            CAPE.XVAE.handler.train.fit(trial=trial)

    if G.TASK.ID == "ls":
        ls_trials(study, ascending=True)
