# coding=utf-8
# Copyright (C) The Zhejiang University KG Lab Authors. team - All Rights Reserved
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
# 
# @FileName  :data_utils.py
# @Time      :2020/11/2 20:08
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# 

import random
import pickle
import logging
from typing import List, Dict, Union, Any

logger = logging.getLogger(__name__)

__all__ = [
    'dummy_data',
    'load_pkl',
    'save_pkl',
]


def dummy_data(num_samples: int = 1000) -> List[Dict[str, Union[List[int], List[List[int]], int]]]:
    """
    生成 num_samples 条数据，每个样本 sent 长度为 [10, 20]，每个 word 长度为 [1, 5]
    0 为 padding, word 级别词典范围 [1, 100], char 级别词典范围 [101, 200]
    有个 label 分类，分类范围在 [0, 9]中十分类。
    有个 tagging 序列标注结果，每个 word 分类在 [0,4]中5分类。

    Args:
        num_samples: 样本数目，默认1000条

    Returns:
       a list
    """
    lengths = [random.randint(10, 20) for _ in range(num_samples)]

    data = []
    for len in lengths:
        data.append({
            'sent': [random.randint(1, 100) for _ in range(len)],
            'word': [[random.randint(101,200) for _ in range(random.randint(1,5))] for _ in range(len)],
            'label': random.randint(0,9),
            'tagging': [random.randint(0,4) for _ in range(len)],
        })

    return data

def load_pkl(fp: str, verbose: bool = True) -> Any:
    if verbose:
        logger.info(f'load data from {fp}')

    with open(fp, 'rb') as f:
        data = pickle.load(f)
        return data


def save_pkl(data: Any, fp: str, vervose: bool =True) -> None:
    if vervose:
        logger.info(f'save data in {fp}')

    with open(fp, 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    pass
