# coding=utf-8
# Copyright (C) The Zhejiang University KG Lab Authors. team - All Rights Reserved
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
# 
# @FileName  :train.py
# @Time      :2020/10/28 22:34
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# Iterative cycle of training process

import os
import sys
import logging
from typing import List, Dict
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, GPUStatsMonitor
from pl_bolts.callbacks import PrintTableMetricsCallback
# self
from dataset import DataModule
from architecture import Model



logging.getLogger("lightning").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # -------------------------------------
    # Randomize seeds during fixed training
    # -------------------------------------
    seed_everything(cfg.seed)
    cfg.cwd = hydra.utils.get_original_cwd()


    # -------------------------------------
    # dataset, build by self
    # -------------------------------------
    datamodule = DataModule(cfg)


    # -------------------------------------
    # model, build by self
    # -------------------------------------
    model = Model(cfg)


    # logger
    tb_logger = pl_loggers.TensorBoardLogger('tensorboard_logs/')

    # callback
    callbacks = []
    if cfg.use_early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=cfg.es_patience, verbose=True))
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    # callbacks.append(PrintTableMetricsCallback())
    # gpu_stats = GPUStatsMonitor()


    # -------------------------------------
    # Other Config, profile, gpus, etc.
    # -------------------------------------
    if cfg.use_gpu and torch.cuda.is_available():
        gpus = cfg.gpus
    else:
        gpus = None

    # -------------------------------------
    # Training start
    # -------------------------------------
    trainer = pl.Trainer(gpus=gpus,
                         profiler=cfg.profiler,
                         precision=cfg.precision,
                         accumulate_grad_batches=cfg.accumulate_grad_batches,
                         flush_logs_every_n_steps=cfg.flush_logs_every_n_steps,
                         log_every_n_steps=cfg.log_every_n_steps,
                         max_epochs=cfg.epochs,  # start from 0 to n-1
                         num_sanity_val_steps=cfg.num_sanity_val_steps,
                         # unit test
                         # fast_dev_run=True
                         terminate_on_nan=True,
                         # accelerator='ddp',
                         logger=[tb_logger],
                         callbacks=callbacks
                         )
    trainer.fit(model=model, datamodule=datamodule)

    # # save to torchscript
    # torch.jit.save(model.to_torchscript(), 'model.pt')
    # os.path.isfile('model.pt')

    # save to onnx
    # with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
    #     autoencoder = LitAutoEncoder()
    #     input_sample = torch.randn((1, 28 * 28))
    #     autoencoder.to_onnx(tmpfile.name, input_sample, export_params=True)
    #     os.path.isfile(tmpfile.name)

    # test or not
    trainer.test(datamodule=datamodule)


if __name__ == '__main__':
    main()

    # hydra.utils.to_absolute_path('/foo')
    # -m



# a.detach()
# t = torch.randn(2, 2, device=self.device)
# self.register_buffer('t', torch.randn(2, 2))
# new_a = torch.Tensor(2, 3).type_as(a)


