# coding=utf-8
# 
# @Version   :3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]
# @Software  :PyCharm
# 
# @FileName  :handle_mini.py
# @Time      :2020/11/11 16:51
# @Author    :Haiyang Yu
# @E-Mail    :yuhaiyang@zju.edu.cn
#
# @Description
# 

import os
import logging
from typing import List, Dict
import json

logger = logging.getLogger(__name__)



def count_relation(fp):
    rels = {}
    with open(fp, encoding='utf-8') as f:
        for l in f:
            data = json.loads(l)
            rel = data['predicate']
            rels[rel] = rels.get(rel, 0) + 1

    rels = sorted(zip(rels.values(), rels.keys()), reverse=True)
    print(rels)

def select_relation(fp, rels):
    train, dev, test = [], [], []
    train_rels, dev_rels, test_rels = {}, {}, {}
    with open(fp, encoding='utf-8') as f:
        for l in f:
            data = json.loads(l)
            rel = data['predicate']

            if rel in rels:
                if train_rels.get(rel, 0) < 100:
                    train_rels[rel] = train_rels.get(rel, 0) + 1
                    train.append(data)
                elif dev_rels.get(rel, 0) < 10:
                    dev_rels[rel] = dev_rels.get(rel, 0) + 1
                    dev.append(data)
                elif test_rels.get(rel, 0) < 10:
                    test_rels[rel] = test_rels.get(rel, 0) + 1
                    test.append(data)
                else:
                    pass

    with open('mtrain.json', 'w', encoding='utf-8') as f:
        f.write(
            os.linesep.join(
                [json.dumps(l, ensure_ascii=False) for l in train]
            )
        )
    with open('mdev.json', 'w', encoding='utf-8') as f:
        f.write(
            os.linesep.join(
                [json.dumps(l, ensure_ascii=False) for l in dev]
            )
        )
    with open('mtest.json', 'w', encoding='utf-8') as f:
        f.write(
            os.linesep.join(
                [json.dumps(l, ensure_ascii=False) for l in test]
            )
        )

if __name__ == '__main__':
    # count_relation('train.json')

    rels = ['作者', '成立日期', '毕业院校', '董事长', '朝代', '主角', '作词', '面积', '气候', '所在城市']
    select_relation('train.json', rels)


    # train
    # [(18867, '作者'), (9893, '成立日期'), (6059, '国籍'), (5850, '毕业院校'), (5561, '歌手'), (5479, '父亲'), (5316, '董事长'),
    #  (5096, '导演'), (3464, '母亲'), (3275, '主演'), (3050, '简称'), (2791, '朝代'), (2685, '校长'), (2236, '上映时间'), (2180, '配音'),
    #  (1782, '主题曲'), (1677, '获奖'), (1642, '嘉宾'), (1351, '出品公司'), (1333, '主持人'), (1264, '主角'), (1204, '总部地点'),
    #  (1189, '票房'), (1170, '祖籍'), (1076, '号'), (1055, '编剧'), (873, '气候'), (827, '创始人'), (814, '所属专辑'), (706, '作词'),
    #  (668, '所在城市'), (623, '代言人'), (530, '作曲'), (461, '占地面积'), (429, '制片人'), (389, '面积'), (368, '海拔'), (338, '首都'),
    #  (196, '人口数量'), (146, '注册资本'), (131, '饰演'), (93, '官方语言'), (23, '邮政编码'), (19, '专业代码'), (13, '修业年限'), (12, '改编自')]



