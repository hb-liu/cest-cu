# define utility functions
import torch
from torch.fft import *
from torch.optim.lr_scheduler import LambdaLR

def ifft2c(k):
    return ifftshift(ifft2(fftshift(k)))

def fft2c(x):
    return fftshift(fft2(ifftshift(x)))

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