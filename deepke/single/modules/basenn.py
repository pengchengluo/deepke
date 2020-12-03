import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNN(nn.Module):
    def __init__(self, cfg):
        super(BaseNN, self).__init__()

        # self.xxx = cfg.xxx
        self.l1out = cfg.l1out
        self.embed = nn.Embedding(num_embeddings=cfg.num_embeddings, embedding_dim=cfg.embedding_dim, padding_idx=0)
        self.l1 = nn.Linear(cfg.embedding_dim, self.l1out)
        self.l2 = nn.Linear(self.l1out, cfg.l2out)

    def forward(self, x):
        embed = self.embed(x)
        x1 = F.relu(self.l1(embed))
        x2 = self.l2(x1)
        return torch.max(x2, dim=1)[0]
