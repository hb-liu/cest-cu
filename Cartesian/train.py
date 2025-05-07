import os
import yaml
import torch
import random
import models
import argparse
import numpy as np
import torch.nn as nn
from utils.utils import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class dataset(Dataset):
    """
    Params:
        - nadj: number of adjacent images to fetch
    """
    def __init__(self, datapath, nadj, databatch):
        super().__init__()
        # load training data according to databatch
        with open(databatch,'r') as f:
            names = f.read().splitlines()
        datadict = {}
        datadict['sat']  = []
        datadict['usat'] = []
        nsamples = 0
        for name in names:
            data = loadmat(os.path.join(datapath,name),simplify_cells=True)['data']
            kspace, offsets = data['kspace'], data['offsets']
            kspace = np.transpose(kspace,[2,0,1])
            datadict['sat'].append(kspace[offsets<100])
            datadict['usat'].append(np.mean(kspace[offsets>=100],0,keepdims=True))
            nsamples += sum(offsets<100)
        self.datadict = datadict
        self.nsamples = nsamples
        self.nadj = nadj
    def __len__(self):
        return self.nsamples
    def __getitem__(self, index):
        # randomly select a cest data
        index = random.randint(0,len(self.datadict['sat'])-1)
        sat   = self.datadict['sat'][index]
        usat  = self.datadict['usat'][index]

        # randomly select a sequence of adjacent images
        index = random.randint(0,len(sat)-self.nadj)
        kf = torch.from_numpy(sat[index:index+self.nadj])
        xf = ifft2c(kf)
        m0 = torch.from_numpy(usat)
        m0 = ifft2c(m0)
        return xf, kf, m0

def main(args):
    # construct fitting model
    model = getattr(getattr(models, args.model), args.model)
    with open(os.path.join('params',args.model+'.yaml'), 'r') as f:
        modelparams = yaml.load(f, yaml.SafeLoader)
    model = model(modelparams).cuda()

    # construct training dataset
    trainset = dataset(args.datapath, modelparams['nadj'], args.databatch)
    trainloader = DataLoader(trainset, modelparams['batch_size'], shuffle=True)

    # run training
    epochs = modelparams['epochs']
    lr, momentum, weight_decay = modelparams['lr'], modelparams['momentum'], modelparams['weight_decay']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = PolyScheduler(optimizer, t_total=epochs)
    criterion = nn.L1Loss()
    iter = 0
    writer = SummaryWriter()
    for epoch in range(epochs):
        for xf, kf, m0 in trainloader:
            ku = model.undersampling(kf).cuda()
            xf, m0 = xf.cuda(), m0.cuda()

            xr = model(ku,m0)
            loss = criterion(xr,xf)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            writer.add_scalar('Loss/train', loss.item(), iter)
            iter = iter + 1
        scheduler.step()
        # plot at every x epoch
        if modelparams['plot']:
            plt.subplot(1,2,1)
            plt.imshow(xr[0][0].real.detach().cpu())
            plt.subplot(1,2,2)
            plt.imshow(xf[0][0].real.detach().cpu())
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        # save at every x epoch
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join('checkpoints',args.model,f'ckp_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='reconstruction model')
    parser.add_argument('--datapath', type=str, help='abs. path of training data')
    parser.add_argument('--databatch', type=str, help='abs. path of databatch file')
    args = parser.parse_args()
    main(args)