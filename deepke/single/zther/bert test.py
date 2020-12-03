import torch
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import numpy as np

fp = 'C:/bert'

config = {
    'num_hidden_layers': 12,
    'add_pooling_layer': True,
}

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', mirror='tuna', cache_dir=fp)

model = BertModel.from_pretrained('bert-base-uncased', mirror='tuna', cache_dir=fp, **config)

t = tokenizer.encode('我是中国人，我爱中国。', return_tensors='pt')

bert_input_config = {
    'output_attentions': True,
    'output_hidden_states': True,
    'return_dict': True,
}
y = model(t, **bert_input_config)
# print(y)


# n 多少层, m 是 多头的一个
n = 11
m = 11


def f1(n=11, m=11):
    ppt = y.attentions[n][0][m].detach().numpy()
    # cmap=plt.cm.YlGnBu,
    plt.imshow(ppt, vmin=0, vmax=1)
    plt.colorbar()
    plt.show()


for i in range(12):
    f1(n=i, m=6)
for j in range(12):
    f1(n=11, m=j)
