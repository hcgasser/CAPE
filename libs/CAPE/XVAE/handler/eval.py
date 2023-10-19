import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib as mpl
import seaborn as sns

import torch
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

import kit
import kit.globals as G
from kit.path import join
from kit.jobs import plot_job_metrics


def run():
    if G.TASK.ID == 'eval_plot_metrics':
        join(G.ENV.ARTEFACTS, "figures")
        if G.TASK.JOB != '' and os.path.exists(G.TASK.PATHS.JOB):
            Ys = G.TASK.Y.split("+")
            for y in Ys:
                filename = join(G.TASK.PATHS.JOB, "figures", f"{G.TASK.JOB}_{y}.pdf")
                plot_job_metrics(filename, G.TASK.JOB, G.TASK.X, y, splits=True,
                                xscale=G.TASK.X_SCALE, yscale=G.TASK.Y_SCALE)
            if not os.path.islink(os.path.join(G.ENV.ARTEFACTS, "figures", G.TASK.JOB)):
                tgt = os.path.join(G.TASK.PATHS.JOB, "figures")
                src = os.path.join(G.ENV.ARTEFACTS, "figures")
                os.symlink(os.path.relpath(tgt, src), os.path.join(src, G.TASK.JOB))
        else:
            raise FileNotFoundError(f"Job '{G.TASK.JOB}' not found.")

def get_data_sequences():
    spl = [kit.Split.TRAIN, kit.Split.VAL, kit.Split.TEST]
    dfs = []
    for split in spl:
        df = kit.bioinf.fasta.read_fasta(os.path.join(G.ENV.INPUT, G.DHPARAMS[str(split)]), return_df=True)
        df['split'] = split.long_name()
        dfs.append(df)

    return pd.concat(dfs, axis=0)

