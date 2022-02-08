import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math


class MixChan(nn.Module):

    def __init__(self, w, pooling_kernel=4):
        super(MixChan, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.mlp = [nn.Linear((w // pooling_kernel) ** 2, (w // pooling_kernel ** 2) ** 2), nn.LeakyReLU(0.2, True),
                    nn.Linear((w // pooling_kernel ** 2) ** 2, 1), nn.LeakyReLU(0.2, True)]

    def forward(self, x, dist):

        dist /= dist.sum()
        x = x.sum(axis=2) * dist.unsqueeze(dim=1).unsqueeze(dim=2)

        n_batch, n_chan, w, h = x.shape
        mixed = torch.zeros((n_batch, 1, w, h))
        down_sampled = self.pooling(x)

        for i in range(n_batch):
            for j, ch in enumerate(down_sampled[i]):
                coef = self.mlp(ch)
                mixed[i][1] += coef * x[i][j]

        return mixed / n_chan
