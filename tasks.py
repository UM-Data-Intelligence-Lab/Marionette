import pytorch_lightning as pl
import torch
import torch.nn as nn
from pathlib import Path
import wandb
import os 

import numpy as np
import matplotlib.pyplot as plt



from datamodule import Batch
from add_thin.metrics import (
    MMD,
    lengths_distribution_wasserstein_distance,
)
from evaluations.statistical_metrics import Get_Statistical_Metrics


class Tasks(pl.LightningModule):
    def __init__(
        self,
        tpp_model,
        discrete_diffusion,
        learning_rate1,
        learning_rate2,
        weight_decay1: float = 0.0,
        weight_decay2: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("tpp_model","discrete_diffusion"))

        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2
        self.learning_rate1 = learning_rate1
        self.learning_rate2 = learning_rate2
        self.tpp_model = tpp_model
        self.discrete_diffusion = discrete_diffusion
        self.classification_loss_func = nn.BCEWithLogitsLoss(reduction="none")

    @property
    def automatic_optimization(self) -> bool:
        return False     

    def classification_loss(self, x_n_int_x_0, x_n: Batch):
        """
        Compute BCE loss for the classification task.
        """
        x_n_int_x_0 = x_n_int_x_0.flatten()[x_n.mask.flatten()]
        target = x_n.kept.flatten()[x_n.mask.flatten()]
        loss = self.classification_loss_func(x_n_int_x_0, target.float())
        loss = (loss).sum() / len(x_n)
        return loss

    def intensity_loss(self, log_prob_x_0):
        """
        Compute the average (over batch) negative log-likelihood of the event sequences.
        """
        return -log_prob_x_0.mean()

    def get_loss(self, log_prob_x_0, x_n_int_x_0, x_n):
        """
        Compute the loss for the classification and intensity.
        """
        intensity = self.intensity_loss(log_prob_x_0) / self.tpp_model.n_max

        classification = (
            self.classification_loss(x_n_int_x_0, x_n) / self.tpp_model.n_max
        )
        loss = classification + intensity
        return loss, classification, intensity

    def step(self, batch, name):
        """
        Apply tpp_model to batch and compute loss.
        """
        # Forward pass
        x_n_int_x_0, log_prob_x_0, x_n = self.tpp_model.forward(batch)

        # Spatial loss
        spatial_loss = self.discrete_diffusion.training_losses(batch).mean()


        # Compute loss
        temporal_loss, classification, intensity = self.get_loss(
            log_prob_x_0, x_n_int_x_0, x_n
        )

        total_loss = spatial_loss + temporal_loss
        # Log loss
        self.log(
            f"{name}/total_loss",
            total_loss.detach().item(),
            batch_size=batch.batch_size,
        )
        self.log(
            f"{name}/spatial_loss",
            spatial_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        self.log(
            f"{name}/temporal_loss",
            temporal_loss.detach().item(),
            batch_size=batch.batch_size,
        )

        self.log(
            f"{name}/log-likelihood",
            intensity.detach().item(),
            batch_size=batch.batch_size,
        )
        if classification is not None:
            self.log(
                f"{name}/BCE",
                classification.detach().item(),
                batch_size=batch.batch_size,
            )
        return temporal_loss,spatial_loss,total_loss

    def configure_optimizers(self):
        optimizer1 = torch.optim.Adam(self.tpp_model.parameters(), lr=self.learning_rate1, weight_decay=self.weight_decay1,)
        
        optimizer2 = torch.optim.AdamW(self.discrete_diffusion.parameters(), lr=self.learning_rate2, weight_decay=self.weight_decay2)

        lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer1, factor=0.95, patience=1000, verbose=True
        )

        lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer2, factor=0.95, patience=1000, verbose=True
        )
        return ({"optimizer": optimizer1,"lr_scheduler": {"scheduler": lr_scheduler1}},
                {"optimizer": optimizer2,"lr_scheduler": {"scheduler": lr_scheduler2}})


class DensityEstimation(Tasks):
    def __init__(
        self, tpp_model, discrete_diffusion, learning_rate1, learning_rate2, weight_decay1, weight_decay2
    ):
        super().__init__(
            tpp_model, discrete_diffusion, learning_rate1, learning_rate2, weight_decay1, weight_decay2
        )

    def training_step(self, batch, batch_idx):
        loss_temporal, loss_spatial, loss_all = self.step(batch, "train")

        opt1, opt2 = self.optimizers()

        opt1.zero_grad()
        self.manual_backward(loss_temporal)
        opt1.step()

        opt2.zero_grad()
        self.manual_backward(loss_spatial)
        opt2.step()

        sch1, sch2 = self.lr_schedulers()
        sch1.step(loss_temporal)
        sch2.step(loss_spatial)

        self.log_dict({"t_loss": loss_temporal, "s_loss": loss_spatial}, prog_bar=True)


    def test_step(self, batch, batch_idx):
        pass

