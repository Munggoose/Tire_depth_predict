import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):

    def __init__(self, patch_size=16, stride=16, padding=0,
                in_chans=3,embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans,embed_dim, kernel_size=patch_size,
                    stride=(stride,stride),padding=(padding,padding))
        self.norm = nn.BatchNorm2d(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class StemConv(nn.Module):

    def __init__(self,in_chs, out_chs):
        super(StemConv,self).__init__()
        self.conv1 = nn.Conv2d(in_chs, out_chs//2, kernerl_size=2, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(out_chs//2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_chs//2, out_chs,kernerl_size=2, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(out_chs)
        self.act2 = nn.ReLU()


    def forward(self, x):
        x = self.act1(self.batch1(self.conv1(x)))
        x = self.act2(self.batch2(self.conv2(x)))
        return x

#  pool_size, stride=1, padding=pool_size // 2,

class MB4D(nn.Module):

    def __init__(self, in_chs, h_chs,out_chs):
        self.pool = nn.AvgPool2d(kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.Conv2d(in_chs,h_chs,stride=1, kernel_size=1)
        self.batch1 = nn.BatchNorm2d(h_chs)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(h_chs,out_chs,stride=2,kernel_size=2)
        self.batch2 = nn.BatchNorm2d(out_chs)
    

    def forward(self, x):
        x_h = self.pool(x) + x
        out = self.batch1(self.conv1(x_h))
        out = self.act(out)
        out =  self.batch2(self.conv2(out))
        out = out + x_h
        return out



class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x