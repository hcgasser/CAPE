from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from kit.data import Split
from kit.log import log_caller


class CapeBasic(pl.LightningModule, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.ar = 0  # should be 1 for autoregressive models

    @abstractmethod
    def step(self, batch, batch_idx, split):
        pass

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, Split.TRAIN)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, Split.VAL)

    def configure_optimizers(self):
        log_caller(self)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.LR,
            weight_decay=self.hparams.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.trainer.max_epochs
        )
        return [optimizer], [scheduler]

    def log_likelihood(self, idx, mask):
        idx = idx.to(self.device)
        mask = mask.to(self.device)

        self.eval()
        all_logits = self(idx)[0]

        ps_dist = F.softmax(all_logits, dim=-1).detach()
        log_ps = ps_dist.gather(-1, idx[:, self.ar :].unsqueeze(dim=-1)).squeeze().log()

        log_ps = log_ps * mask[:, self.ar :]
        log_p = log_ps.sum(dim=-1)

        return log_p.to("cpu"), log_ps.to("cpu")


class CapeVAE(CapeBasic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.beta = 0.0

        self.Zs = []
        self.mu_hat = torch.nn.Parameter(None)
        self.cov_hat = torch.nn.Parameter(None)

    @abstractmethod
    def encode(self, idx):
        return None

    @abstractmethod
    def sample_latents(self, N):
        pass

    @abstractmethod
    def save_decoder(self):
        pass

    @abstractmethod
    def reset_decoder(self):
        pass

    def anneal_beta(self):
        if self.current_epoch < self.hparams.EPOCHS_AE:
            return 0.0

        if self.current_epoch < (self.hparams.EPOCHS_AE + self.hparams.EPOCHS_ANNEAL):
            return (
                self.current_epoch - self.hparams.EPOCHS_AE
            ) / self.hparams.EPOCHS_ANNEAL

        return 1.0

    def step(self, batch, batch_idx, split):
        x = batch["x"]["seq"]
        weight = batch["x"]["weight"]

        N, T_in = x.shape
        T = T_in - self.ar

        y_hat, z, p, q = self.forward(x)[:4]
        # y_hat = self.forward(x)

        self.Zs.append(z.detach().to("cpu"))

        loss_recon_unweighted = F.cross_entropy(
            y_hat.reshape(N * T, self.hparams.N_TOKENS),
            x[:, self.ar :].reshape(N * T),
            reduction="none",
        )
        # kit.pdb.set_trace()
        # weight = weight.reshape(N, 1).expand(N, T).reshape(N * T)
        loss_recon = loss_recon_unweighted.reshape(N, T)
        loss_recon = loss_recon.sum(-1)  # sum over sequence
        loss_recon = loss_recon * weight  # weight by frequency of sequence
        loss_recon = loss_recon.mean()  # average over batch samples

        loss_kl = torch.distributions.kl_divergence(q, p)
        loss_kl_hinge = torch.maximum(
            loss_kl, torch.tensor(self.hparams.LAMBDA, device=self.device)
        )  # hinge loss
        while len(loss_kl.shape) > 1:  # sum over all latent dimensions
            loss_kl_hinge = loss_kl_hinge.sum(-1)
            loss_kl = loss_kl.sum(-1)
        loss_kl_hinge = loss_kl_hinge * weight  # weight by frequency of the sequence
        loss_kl = loss_kl * weight
        loss_kl_hinge = loss_kl_hinge.mean()  # average over batch samples
        loss_kl = loss_kl.mean()

        beta = self.anneal_beta()
        loss = loss_recon + loss_kl_hinge * beta

        self.log("beta", beta, prog_bar=True, logger=True, rank_zero_only=True)

        self.log(
            f"{split}_loss", loss, prog_bar=False, logger=True, rank_zero_only=True
        )
        self.log(
            f"{split}_loss_recon",
            loss_recon,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
        )
        self.log(
            f"{split}_loss_kl", loss_kl, prog_bar=True, logger=True, rank_zero_only=True
        )

        return loss

    def on_train_epoch_start(self):
        self.Zs = []
        if self.current_epoch == 0:
            self.save_decoder()
        elif self.current_epoch == self.hparams.EPOCHS_AE:
            self.reset_decoder()

    def on_train_epoch_end(self):
        if len(self.Zs) > 0:
            Zs = torch.concat([z.to(self.device) for z in self.Zs], dim=0)
            if len(Zs.shape) == 3:
                Zs = Zs.permute(1, 2, 0)
                mu_hat, cov_hat = [], []
                for Z in Zs:
                    mu_hat.append(Z.mean(dim=1))
                    cov_hat.append(Z.cov())

                self.mu_hat = torch.nn.Parameter(torch.stack(mu_hat))
                self.cov_hat = torch.nn.Parameter(torch.stack(cov_hat))

            elif len(Zs.shape) == 2:
                Z = Zs.permute(1, 0)
                self.mu_hat = torch.nn.Parameter(Z.mean(dim=1))
                self.cov_hat = torch.nn.Parameter(Z.cov())
