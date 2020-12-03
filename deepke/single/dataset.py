# coding=utf-8
# Copyright (C) The Zhejiang University KG Lab Authors. team - All Rights Reserved
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
#
# @FileName  :dataset.py
# @Time      :2020/10/28 22:34
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# Load dataset and build dataloader

import os
import csv
import json
from typing import Optional, Union, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from transformers import BertTokenizer
# self
from utils import load_pkl, save_pkl, dummy_data


class DataModule(pl.LightningDataModule):
    """
    Load dataset and build dataloader.

    Attributes:
        cfg.data_dir (str): original data file path
    """
    def __init__(self, cfg):
        super().__init__()
        # self.xxx = cfg.xxx
        self.cwd = cfg.cwd
        self.data_dir = cfg.data_dir
        self.has_preprocessed_data = cfg.has_preprocessed_data
        self.save_preprocessed_data = cfg.save_preprocessed_data

        # bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(cfg.bert_type, cache_dir=cfg.cache_dir, mirror=cfg.mirror)

        # relation config
        self.add_special_indicator = cfg.add_special_indicator
        self.replace_entity_to_type = cfg.replace_entity_to_type

        # dataloader
        self.batch_size = cfg.batch_size
        self.dataloader_num_workers = cfg.dataloader.num_workers
        self.dataloader_pin_memory = cfg.dataloader.pin_memory


    def _hanle_relation_sent(self, fp: str, rels, rel_head, rel_tail):
        output_data = []
        with open(fp, encoding='utf-8') as f:
            for l in f:
                d = json.loads(l)
                data = {}
                text = d['text']
                if self.add_special_indicator:
                    text = text \
                        .replace(d['subject'], '<head_start>' + d['subject'] + '<head_end>') \
                        .replace(d['object'], '<tail_start>' + d['object'] + '<tail_end>')
                if self.replace_entity_to_type:
                    text = text \
                        .replace(d['subject'], rel_head[d['predicate']]) \
                        .replace(d['object'], rel_tail[d['predicate']])
                data['x'] = self.tokenizer.encode(text)
                data['y'] = rels[d['predicate']]
                output_data.append(data)
        return output_data


    def prepare_data(self) -> None:
        """
        OPTIONAL, define functions such as how to download(), tokenize, etc…
        called only on 1 GPU/machine
        """

        # handle relation schema
        rels, rel_head, rel_tail = {}, {}, {}
        with open(os.path.join(self.cwd, self.data_dir, 'schema.csv'), encoding='utf-8') as f:
            writer = csv.DictReader(f, fieldnames=None, dialect='excel')
            for line in writer:
                rels[line['predicate']] = line['index']
                rel_head[line['predicate']] = line['subject_type']
                rel_tail[line['predicate']] = line['object_type']

        # special token for entity pairs in text
        self.tokenizer.add_special_tokens({'additional_special_tokens':["<head_start>", "<head_end>", "<tail_start>", "<tail_end>"]})

        if self.has_preprocessed_data:
            self.train_data = load_pkl(os.path.join(self.cwd, self.data_dir, 'train.pkl'))
            self.dev_data = load_pkl(os.path.join(self.cwd, self.data_dir, 'dev.pkl'))
            self.test_data = load_pkl(os.path.join(self.cwd, self.data_dir, 'test.pkl'))
        else:
            self.train_data = self._hanle_relation_sent(os.path.join(self.cwd, self.data_dir, 'train.json'), rels, rel_head, rel_tail)
            self.dev_data = self._hanle_relation_sent(os.path.join(self.cwd, self.data_dir, 'dev.json'), rels, rel_head, rel_tail)
            self.test_data = self._hanle_relation_sent(os.path.join(self.cwd, self.data_dir, 'test.json'), rels, rel_head, rel_tail)

        if self.save_preprocessed_data and not self.has_preprocessed_data:
            save_pkl(self.train_data, os.path.join(self.cwd, self.data_dir, 'train.pkl'))
            save_pkl(self.dev_data, os.path.join(self.cwd, self.data_dir, 'dev.pkl'))
            save_pkl(self.test_data, os.path.join(self.cwd, self.data_dir, 'test.pkl'))


    def setup(self, stage: Optional[str] = None):
        """
        # OPTIONAL, how to split, etc…
        called for every GPU/machine (assigning state is OK)

        Args:
            stage:

        Returns:

        """
        self.train = self.train_data
        self.val = self.dev_data
        self.test = self.test_data


    def _collate_fn(self, batch):
        """
        The collate function is used to make one batch data meet the needs of model input.

        Args:
            batch (List): one batch data from dataset

        Returns:

        """
        batch = sorted(batch, key=lambda x: len(x['x']), reverse=True)
        max_len = len(batch[0]['x'])

        x, x_len, y = [], [], []
        for sample in batch:
            len_hat = len(sample['x'])
            x_len.append(len_hat)
            x.append(sample['x'] + [0] * (max_len - len_hat))
            y.append(int(sample['y']))

        x, x_len, y = torch.tensor(x), torch.tensor(x_len), torch.tensor(y)

        return x, x_len, y

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory,
                          collate_fn=self._collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory,
                          collate_fn=self._collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.dataloader_num_workers,
                          pin_memory=self.dataloader_pin_memory,
                          collate_fn=self._collate_fn)
