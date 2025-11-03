# define utility functions
import torch
from math import sqrt
from torch.fft import *
from torch.optim.lr_scheduler import LambdaLR

def ifft2c(k):
    return ifftshift(ifft2(fftshift(k))) * sqrt(k.shape[-2]*k.shape[-1])

def fft2c(x):
    return fftshift(fft2(ifftshift(x))) / sqrt(x.shape[-2]*x.shape[-1])

def view_as_complex(x):
    x = torch.chunk(x, chunks=2, dim=1)
    x = x[0] + 1j * x[1]
    return x

def view_as_real(x):
    x = torch.cat([x.real, x.imag], dim=1).float()
    return x

class PolyScheduler(LambdaLR):
    def __init__(self, optimizer, t_total, exponent=0.9, last_epoch=-1):
        self.t_total = t_total
        self.exponent = exponent
        super(PolyScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return (1 - step / self.t_total)**self.exponent
    
def sense(img, csm) -> Tensor:
    """
    calculate coil-combined images
    Args:
        img: multi-coil images, `(batchsize, ncoils, noffs, height, width)`
        csm: coil sensitivity maps, `(batchsize, ncoils, noffs, height, width)`
    Returns:
        rec: reconstructed coil-combined images, `(batchsize, noffs, height, width)`
    """
    rec = torch.sum(img * torch.conj(csm), dim=1)
    return rec