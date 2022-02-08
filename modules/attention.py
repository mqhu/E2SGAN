import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, math

'''借鉴http://nlp.seas.harvard.edu/2018/04/03/attention.html'''
class MutualAttention(nn.Module):

    def __init__(self, w, h):
        super(MutualAttention, self).__init__()
        self.w = w
        self.h = h
        # self.hidde_w = hidde_w
        self.linears = clones(nn.Linear(w, w), 2)
        # self.linear = nn.Linear(hidde_w, w)
        self.attn = None

    def forward(self, query, key):
        nbatches = query.size(0)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        # query, key, value = [query.view(nbatches, -1, self.w * self.h), key.view(nbatches, -1, self.w * self.h),
        #                      value.view(nbatches, -1, self.w * self.h)]
        x1, self.attn1 = attention(query, key, query)
        x2, self.attn2 = attention(key, query, key)
        # x = self.linears[-1](x)  # 附加一个线性层，因为初始有个线性层，想对称一下把空间变回去
        # x = self.linear(x)
        # x = x.view(nbatches, -1, self.w, self.h)
        return torch.cat((x1, x2), dim=1)


class OneSidedAttention(nn.Module):

    def __init__(self, w, h):
        super(OneSidedAttention, self).__init__()
        self.w = w
        self.h = h
        # self.hidde_w = hidde_w
        self.linears = clones(nn.Linear(w, w), 2)
        # self.linear = nn.Linear(hidde_w, w)
        self.attn = None
        self.query = None
        self.key = None

    def forward(self, query, key):
        nbatches = query.size(0)
        query, key = [l(x) for l, x in zip(self.linears, (query, key))]
        # query, key, value = [query.view(nbatches, -1, self.w * self.h), key.view(nbatches, -1, self.w * self.h),
        #                      value.view(nbatches, -1, self.w * self.h)]
        x1, self.attn = attention(query, key, query)
        self.query = query
        self.key = key
        # x = self.linears[-1](x)  # 附加一个线性层，因为初始有个线性层，想对称一下把空间变回去
        # x = self.linear(x)
        # x = x.view(nbatches, -1, self.w, self.h)
        return x1


def attention(query, key, value):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
