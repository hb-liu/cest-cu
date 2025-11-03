# define multi-offset transformer reconstruction network
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from utils.utils import *
import torchkbnufft as tkbn
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
    def forward(self, x, xj):
        x  = view_as_real(x)
        xj = view_as_real(xj)
        x  = torch.cat([x,xj],dim=1)
        x  = self.convin(x)
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = self.convout(x)
        out = view_as_complex(out)
        return out

class MoTR(nn.Module):
    def __init__(self, modelparams: dict):
        super().__init__()
        rate   = modelparams['rate']
        nadj   = modelparams['nadj']
        hdim   = modelparams['hdim']
        niter  = modelparams['niter']
        imsize = tuple(modelparams['imsize'])
        self.iters = nn.ModuleList()
        for i in range(niter):
            self.iters.append(Transformer(2*(1+nadj), hdim, 2*nadj))
        self.nadj   = nadj
        self.rate   = rate
        self.imsize = imsize
        self.nufft  = tkbn.KbNufft(imsize)
        self.adj    = tkbn.KbNufftAdjoint(imsize)
        self.dcf    = None
        self.jdcf   = None
    @torch.no_grad()
    def join(self, ku: Tensor, tu: Tensor) -> Tensor:
        """
        join adjacent frequency offsets to form fully-sampled reference images
        Args:
            ku: undersampled k-space, `(batch_size, nadj, traj_length, nleaves)`
            tu: undersampling trajectory, `(nadj, ndim, traj_length, nleaves)`
        Returns:
            xj: reference images, `(batch_size, 1, height, width)`
        """
        ku, tu = ku.flatten(2), tu.flatten(2)
        ku = rearrange(ku, 'b n l -> b 1 (n l)')
        tu = rearrange(tu, 'n d l -> 1 d (n l)')
        if self.jdcf == None:
            self.jdcf = self.update_dcf(tu)
        batchsize = ku.shape[0]
        tu  = tu.repeat([batchsize,1,1])
        dcf = self.jdcf.repeat([batchsize,1,1])
        ku  = rearrange(ku, 'b n l -> (b n) 1 l')
        xu  = self.adj(ku*dcf, tu, norm='ortho')
        return xu
    @torch.no_grad()
    def undersampling(self, kf: Tensor, traj: Tensor) -> Tensor:
        """
        perform retrospective undersampling of fullysampled image
        Args:
            kf: fullysampled k-space, `(batch_size, nadj, traj_length, nleaves)`
            traj: fullysampling trajectory, `(nadj, ndim, traj_length, nleaves)`
        Returns:
            ku,tu: 
            undersampled k-space, `(batch_size, nadj, traj_length, nleaves)`
            undersampling trajectory, `(nadj, ndim, traj_length, nleaves)`
        """
        nleaves = traj.shape[-1]
        # ku has fewer leaves, also complements each other
        ku = torch.zeros(kf.shape[0:3]+(nleaves//self.rate,),dtype=kf.dtype).to(kf.device)
        tu = torch.zeros(traj.shape[0:3]+(nleaves//self.rate,),dtype=traj.dtype).to(traj.device)
        for i in range(self.nadj):
            # if kf has 6 leaves, indexed as [0,1,2,3,4,5]
            # and undersampling rate is 3, then each ku gets
            # 2 leaves, indexed [0,3],[1,4],[2,5], respectively
            idx = list(range(i,nleaves,self.rate))
            ku[:,i] = kf[:,i,:,idx]
            tu[i] = traj[i,:,:,idx]
        return ku, tu
    @torch.no_grad()
    def update_dcf(self, tu: Tensor):
        dcf = tkbn.calc_density_compensation_function(tu, self.imsize)
        return dcf
    def batchwise_adj(self, ku: Tensor, tu: Tensor) -> Tensor:
        """
        for each sample in minibatch, reconstruct zero-filled image
        Args:
            ku: undersampled k-space, `(batch_size, nadj, traj_length)`
            tu: undersampling trajectory, `(nadj, ndim, traj_length)`
        Returns:
            xu: zero-filled image, `(batch_size, nadj, height, width)`
        """
        batchsize = ku.shape[0]
        tu  = tu.repeat([batchsize,1,1])
        dcf = self.dcf.repeat([batchsize,1,1])
        ku  = rearrange(ku, 'b n l -> (b n) 1 l')
        xu  = self.adj(ku*dcf, tu, norm='ortho').squeeze(1)
        xu  = rearrange(xu, '(b n) h w -> b n h w', b = batchsize)
        return xu
    def batchwise_nufft(self, x: Tensor, traj: Tensor) -> Tensor:
        """
        for each sample in minibatch, interpolate k-space points on trajectory
        Args:
            x: image, `(batch_size, nadj, height, width)`
            traj: sampling trajectory, `(nadj, ndim, traj_length)`
        Returns:
            k: interpolated k-space points, `(batchsize, nadj, traj_length)`
        """
        batchsize = x.shape[0]
        traj = traj.repeat([batchsize,1,1])
        x = rearrange(x, 'b n h w -> (b n) 1 h w')
        k = self.nufft(x, traj, norm='ortho').squeeze(1)
        k = rearrange(k, '(b n) l -> b n l', b = batchsize)
        return k
    def forward(self, ku: Tensor, tu: Tensor, xj: Tensor) -> Tensor:
        """
        Args:
            ku: undersampled k-space, `(batch_size, nadj, traj_length, nleaves)`
            tu: undersampling trajectory, `(nadj, ndim, traj_length, nleaves)`
            xj: reference image by joining adjacent k-space, `(batch_size, 1, height, width)`
        Returns:
            x: reconstructed image, `(batch_size, nadj, height, width)`
        """
        ku, tu = ku.flatten(2), tu.flatten(2)
        if self.dcf == None:
            self.dcf = self.update_dcf(tu)
        x = self.batchwise_adj(ku,tu)
        for layer in self.iters[:-1]:
            x = x + layer(x,xj)
            # data consistency
            kd = self.batchwise_nufft(x,tu) - ku
            xd = self.batchwise_adj(kd,tu)
            x  = x - xd
        # no data consistency at final layer, to remove aliasing artifacts
        x = x + layer(x,xj)
        return x