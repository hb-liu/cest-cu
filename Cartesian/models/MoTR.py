# define multi-offset transformer reconstruction network
import torch
import numpy as np
import torch.nn as nn
from utils.utils import *
from einops import rearrange
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)
        self.dim = dim
    def forward(self, x_in):
        b, c, h, w = x_in.shape
        x = x_in.permute(0, 2, 3, 1).reshape(b,h*w,c)
        # b, hw, hd
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # b, h, hw, d
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q_inp, k_inp, v_inp))
        # b, h, d, hw
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        # attn: b, h, d, d
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        # x: b, h, d, hw
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        # out: b, c, h, w
        out_c = self.proj(x).view(b, h, w, c).permute(0, 3, 1, 2)
        out_p = self.pos_emb(x_in)
        out = out_c + out_p
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            nn.GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )
    def forward(self, x):
        return self.net(x)

class Transformer(nn.Module):
    def __init__(self, idim, hdim, odim, dim_head=64, heads=8, num_blocks=2):
        super().__init__()
        self.convin = nn.Sequential(nn.Conv2d(idim, hdim, 3, 1, 1, bias=False), nn.GELU())
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList([
                    Attention(dim=hdim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=hdim)
                ])
            )
        self.convout = nn.Conv2d(hdim, odim, 1, 1, 0)
    def forward(self, x, m0):
        m0 = view_as_real(m0)
        x  = view_as_real(x)
        x  = torch.cat([x,m0],1)
        x  = self.convin(x)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = self.convout(x)
        out = view_as_complex(out)
        return out

class MoTR(nn.Module):
    def __init__(self, modelparams):
        super().__init__()
        rate  = modelparams['rate']
        nadj  = modelparams['nadj']
        hdim  = modelparams['hdim']
        niter = modelparams['niter'] 
        self.iters = nn.ModuleList()
        for i in range(niter):
            self.iters.append(Transformer(2*(nadj+1), hdim, 2*nadj))
        self.rate = rate
        self.nadj = nadj
    @torch.no_grad()
    def undersampling(self, kf) -> Tensor:
        """
        perform retrospective undersampling of fullysampled image
        Args:
            kf: fullysampled k-space, `(batchsize, nadj, height, width)`
        Returns:
            ku: undersampled k-space, `(batchsize, nadj, height, width)`
        """
        # number of center/peripheral kspace to be sampled
        nce = npe = kf.shape[2]//self.rate//2
        # index of center/peripheral kspace
        c = list(range((kf.shape[2]-nce)//2, (kf.shape[2]+nce)//2))
        p = np.setdiff1d(list(range(kf.shape[2])), c)
        # sampling
        np.random.shuffle(p)
        p = np.split(p, self.nadj)
        ku = torch.zeros_like(kf)
        for i in range(self.nadj):
            ku[:,i,c] = kf[:,i,c]
            ku[:,i,p[i]] = kf[:,i,p[i]]
        return ku
    def forward(self, ku, m0) -> Tensor:
        """
        Args:
            ku: undersampled k-space, `(batchsize, nadj, height, width)`
            m0: fullysampled, unsaturated image, `(batchsize, 1, height, width)`
        Returns:
            x: reconstructed image, `(batchsize, nadj, height, width)`
        """
        x = ifft2c(ku)
        for layer in self.iters:
            x = x + layer(x,m0)
            # data consistency
            kdc = fft2c(x) * (ku == 0) + ku
            x = ifft2c(kdc)
        return x