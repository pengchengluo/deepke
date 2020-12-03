# coding=utf-8
# 
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
# 
# @FileName  :nn_utils.py
# @Time      :2020/11/11 21:36
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# 


import logging
import torch
import numpy as np
from typing import List, Dict, Union

logger = logging.getLogger(__name__)


def seq_len_to_mask(seq_len: Union[List, np.ndarray, torch.Tensor], max_len=None, mask_pos_to_true=True) -> Union[List, np.ndarray, torch.Tensor]:
    """
    将一个表示sequence length的一维数组转换为二维的mask，默认pad的位置为 0。

    Args:
        seq_len: shape是 (B,)
        max_len: 将长度pad到这个长度。默认(None)使用的是seq_len中最长的长度。但在nn.DataParallel的场景下可能不同卡的seq_len会有区别，所以需要传入一个max_len使得mask的长度是pad到该长度。
        mask_pos_to_true: 为 True 时，pad 的部分（0） 为 1， False 则 mask 的部分为 0

    Returns: shape将是(B, max_length)元素类似为 bool 或 torch.uint8
    """
    if isinstance(seq_len, list):
        seq_len = np.array(seq_len)

    if isinstance(seq_len, np.ndarray):
        seq_len = torch.from_numpy(seq_len)

    if isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, logger.error(f"seq_len can only have one dimension, got {seq_len.dim()} != 1.")
        batch_size = seq_len.size(0)
        max_len = int(max_len) if max_len else seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len.device)
        if mask_pos_to_true:
            mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
        else:
            mask = broad_cast_seq_len.ge(seq_len.unsqueeze(1))
    else:
        raise logger.error("Only support 1-d list or 1-d numpy.ndarray or 1-d torch.Tensor.")

    return mask


if __name__ == '__main__':
    pass
