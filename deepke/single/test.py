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
import logging
from typing import List, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pytorch_lightning as pl
# self
from dataset import DataModule
from architecture import Model


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # print config for logger
    cfg.cwd = hydra.utils.get_original_cwd()
    config_msg = '*' * 10 + ' configuration ' + '*' * 10 + '\n' + OmegaConf.to_yaml(cfg)
    logger.info(config_msg)
    logger.info('*' * 35)

    # dataset
    datamodule = DataModule(cfg)

    # model
    model = Model(cfg)

    # load ckpt
    model = model.load_from_checkpoint(cfg.ckpt_path)

    # train
    # trainer = pl.Trainer(profiler=False,
    #                      gpus=cfg.gpus,
    #                      precision=cfg.precision,
    #                      accumulate_grad_batches=cfg.accumulate_grad_batches,
    #                      log_every_n_steps=cfg.log_every_n_steps,
    #                      max_epochs=cfg.epochs,
    #                      # unit test
    #                      # fast_dev_run=True
    #                      )

    trainer = pl.Trainer()

    # test or not
    trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    main()

    # hydra.utils.to_absolute_path('/foo')
    # -m


# pl best formance
#
#
# # use
# a.detach()
# t = torch.randn(2, 2, device=self.device)
# self.register_buffer('t', torch.randn(2, 2))
# new_a = torch.Tensor(2, 3).type_as(a)