""" this file contains the handler for the train task """

from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

import kit
import kit.globals as G
import kit.path
from kit.jobs import get_job_info
from kit.data import Split


def run():
    """main entrance into the handler"""

    metric = "VAL_loss" if "METRIC" not in G.TASK else G.TASK.METRIC
    result = fit(metric=metric)
    print(f"Final {metric}: {result}")


def fit(trial=None, metric="VAL_loss"):
    """function to train the CAPE-XVAE model

    The same function is used for both the training and the continuation runs.

    :param trial: optinal parameter specifying the optuna trial object for newly created runs
        the MHPARAMS and DHPARAMS files will contain functionality to sample hyper-parameters
    :param metric: the metric to return
    :returns: the metric value
    """

    ckpt_path = None
    if G.TASK.ID != "train_continue":
        # if this is not a continuation run, load the hyperparameters and save them
        kit.load_mhparams(G.TASK.MHPARAMS, trial=trial)
        kit.load_dhparams(G.TASK.DHPARAMS, trial=trial)

        G.MHPARAMS.to_yaml(kit.path.join(G.ENV.JOB, "MHPARAMS.yaml"), overwrite=False)
        G.DHPARAMS.to_yaml(kit.path.join(G.ENV.JOB, "DHPARAMS.yaml"), overwrite=False)
    else:
        # else, find the original hyper-parameters and the checkpoint
        job_info = get_job_info(
            G.TASK.JOB_TO_CONTINUE, mhparams=True, dhparams=True, ckpt="last"
        )
        G.MHPARAMS, G.DHPARAMS, (_, ckpt_path) = (
            job_info[k] for k in ["mhparams", "dhparams", "ckpt"]
        )

        if G.DHPARAMS:
            with open(
                kit.path.join(
                    G.ENV.JOBS,
                    G.TASK.JOB_TO_CONTINUE,
                    "continued",
                    f"continued-{G.ENV.ID}.log",
                ),
                "a",
            ) as file:
                file.write(f"{G.JOB.ID}")

    # dynamically initialize the model class
    if ckpt_path is None:
        kit.MODEL = G.MHPARAMS.CLS(**G.MHPARAMS)
    else:
        kit.MODEL = G.MHPARAMS.CLS.load_from_checkpoint(ckpt_path)

    # dynamically initialize the dataset
    ds, dl = {}, {}
    for split in [Split.TRAIN, Split.VAL]:
        ds[split] = G.DHPARAMS.CLS(split, G.ENV.INPUT, G.DHPARAMS)
        dl[split] = DataLoader(
            ds[split],
            batch_size=G.MHPARAMS.BATCH_SIZE,
            shuffle=(split == Split.TRAIN),
            num_workers=G.ENV.CONFIGS.NUM_WORKERS,
            collate_fn=ds[split].collate,
        )

    pl_logger = CSVLogger(G.ENV.JOBS, name="", version=G.JOB.ID)

    best_checkpoint = ModelCheckpoint(
        monitor=f"{Split.VAL}_loss",
        mode="min",
        every_n_epochs=1,
        save_top_k=1,
        filename=f"best_{Split.VAL}_loss",
    )
    latest_checkpoint = ModelCheckpoint(
        monitor="step", mode="max", every_n_epochs=1, save_top_k=1, filename="last"
    )
    callbacks = [
        TQDMProgressBar(refresh_rate=G.ENV.CONFIGS.PB),
        best_checkpoint,
        latest_checkpoint,
    ]

    trainer = Trainer(
        max_epochs=G.TASK.MAX_EPOCHS,
        logger=pl_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accelerator=kit.DEVICE,
    )

    if ckpt_path is None:
        trainer.fit(
            model=kit.MODEL,
            train_dataloaders=dl[Split.TRAIN],
            val_dataloaders=dl[Split.VAL],
        )
    else:
        trainer.fit(
            model=kit.MODEL,
            train_dataloaders=dl[Split.TRAIN],
            val_dataloaders=dl[Split.VAL],
            ckpt_path=ckpt_path,
        )

    return float(trainer.logged_metrics[metric])
