# coding=utf-8
# Copyright (C) The Zhejiang University KG Lab Authors. team - All Rights Reserved
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
# 
# @FileName  :architecture.py
# @Time      :2020/10/29 16:13
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# pytorch_lightning main model class architecture


import logging
from typing import List, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional as FM
from torch import optim
# self
from modules import LmModel


py_logger = logging.getLogger(__name__)


class Model(pl.LightningModule):
    """
    Main model loop contains train/valid/test step/batch loop, and optimizer/loss

    Attributes:
        cfg._model (nn.Module): main module structure
    """

    def __init__(self, cfg):
        super().__init__()
        self._model = LmModel(cfg)
        self.save_hyperparameters(cfg)


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    def _share_step(self, batch, batch_idx):
        """
        Train and valid shared cycle process

        Args:
            batch (Dict):
            batch_idx (int):

        Returns:


        """
        x, x_len, y = batch
        y_hat = self._model(x, x_len)

        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)

        return loss, acc


    def training_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch, batch_idx)
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(metrics,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return metrics


    def validation_step(self, batch, batch_idx):
        loss, acc = self._share_step(batch, batch_idx)
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return metrics

    def forward(self, *batch: Dict):
        x, x_len = batch
        y_hat = self._model(x, x_len)

        return torch.max(y_hat, dim=-1)[1]

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, x_len, y = batch
        y_hat = self.forward(x, x_len)

        acc = torch.sum(y == y_hat) / (y.shape[0] * 1.0)

        self.log('test_acc', acc.detach().item())


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = {'scheduler': optim.lr_scheduler.ExponentialLR(optimizer, 0.3),
                        'name': 'learning_rate'}

        return [optimizer], [lr_scheduler]



if __name__ == '__main__':
    pass



# def training_step():
    # backward acts like normal backward
    # self.manual_backward(loss, opt_a, retain_graph=True)
    # self.manual_backward(loss, opt_a)
    # opt_a.step()
    # opt_a.zero_grad()