# coding=utf-8
# 
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
# 
# @FileName  :lm.py
# @Time      :2020/11/11 21:23
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# 


import logging
from typing import List, Dict
from transformers import BertModel
from torch import nn
logger = logging.getLogger(__name__)
# self
from utils import seq_len_to_mask

class LmModel(nn.Module):
    def __init__(self, cfg):
        super(LmModel, self).__init__()

        bert_model_cfg = {'num_hidden_layers': 4}
        self.bert = BertModel.from_pretrained(cfg.bert_type, cache_dir=cfg.cache_dir, mirror=cfg.mirror, **bert_model_cfg)
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 4)

    def forward(self, x, x_len):
        attention_mask = seq_len_to_mask(x_len)
        outputs = self.bert(x, attention_mask=attention_mask, return_dict=True)
        return outputs['pooler_output']


if __name__ == '__main__':
    pass
