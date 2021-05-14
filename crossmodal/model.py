from functools import partial
import glob
import os
import random


from more_itertools import flatten
import numpy as np
import pytorch_lightning as pl
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(torch.nn.Module):
    def __init__(self, input1_dim, input2_dim, middle_dim, output_dim):
        super().__init__()
        self.linear1_1 = torch.nn.Linear(input1_dim, middle_dim, bias=True)
        self.batch_norm1_1 = torch.nn.BatchNorm1d(middle_dim)
        self.linear1_2 = torch.nn.Linear(middle_dim, output_dim, bias=True)
        self.linear2_1 = torch.nn.Linear(input2_dim, middle_dim, bias=True)
        self.batch_norm2_1 = torch.nn.BatchNorm1d(middle_dim)
        self.linear2_2 = torch.nn.Linear(middle_dim, output_dim, bias=True)

    def forward(self, input1, input2):
        input1 = torch.nn.functional.elu(self.batch_norm1_1(self.linear1_1(input1)))
        input1 = self.linear1_2(input1)
        input2 = torch.nn.functional.elu(self.batch_norm2_1(self.linear2_1(input2)))
        input2 = self.linear2_2(input2)
        output1 = input1 / torch.norm(input1, dim=1).reshape((-1, 1))
        output2 = input2 / torch.norm(input2, dim=1).reshape((-1, 1))
        return output1, output2


class ContrastiveLoss(torch.nn.modules.loss._Loss):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        n = output1.shape[0]

        D = output1.matmul(output2.T)
        d = torch.diag(D).reshape((-1, 1))

        M = torch.eye(n)
        O = D[(M <= 0)].reshape((n, n-1))

        L = self.margin - d
        losses = torch.clamp(L.repeat(1, n-1) + O, 0, 1000)
        return losses.mean()


class Baseline(pl.LightningModule):

    def __init__(self, hparams):
        super(Baseline, self).__init__()
        self.hparams = hparams
        self.model = MLP(hparams.input1_dim, hparams.input2_dim, hparams.middle_dim, hparams.output_dim)
        self.criterion = ContrastiveLoss(margin=1.0)

    def forward(self, input1, input2):
        return self.model.to(device)(input1.to(device), input2.to(device))

    def step(self, batch, batch_idx):
        input1, input2 = batch
        output1, output2 = self.forward(input1, input2)
        return self.criterion(output1, output2)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {'loss': loss,
                'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return {'val_loss': loss,
                'log': {'val_loss': loss}}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss,
                'log': {'val_loss': avg_loss},
                'progress_bar': {'val_loss': avg_loss}}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=3,
                                                               min_lr=1e-6,
                                                               verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
