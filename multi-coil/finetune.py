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
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

class dataset(Dataset):
    """
    Params:
        - nadj: number of adjacent images to fetch
    """
    def __init__(self, datapath, csmpath, nadj):
        super().__init__()
        # load training data according to databatch
        with open('databatch/trains.txt','r') as f:
            names = f.read().splitlines()
        datadict = {}
        datadict['sat']  = {}
        datadict['sat']['kf'] = []
        datadict['sat']['cb'] = []
        datadict['usat'] = []
        datadict['csm']  = []
        nsamples = 0
        for name in names:
            kspace = loadmat(os.path.join(datapath,name),simplify_cells=True)['data']
            csm = loadmat(os.path.join(csmpath,name),simplify_cells=True)['csm']
            ncoils, noffs = kspace.shape[2], kspace.shape[3]
            kspace = np.transpose(kspace,[2,3,0,1])
            csm = np.transpose(csm,[2,3,0,1])
            csm = torch.from_numpy(csm[:,0:1])
            # coil combine
            sat  = torch.from_numpy(kspace[:,1:,:,:])
            usat = torch.from_numpy(kspace[:,0:1,:,:])
            cb_usat = sense(usat[None], csm[None])[0]
            cb_sat  = sense(sat[None], csm[None])[0]
            # store
            datadict['usat'].append(cb_usat)                        # (1, height, width)
            datadict['csm'].append(csm)                             # (ncoils, 1, height, width)
            datadict['sat']['kf'].append(sat)                       # (ncoils, noffs, height, width)
            datadict['sat']['cb'].append(cb_sat)                    # (noffs, height, width)
            nsamples += noffs
        self.datadict = datadict
        self.nsamples = nsamples
        self.nadj = nadj
    def __len__(self):
        return self.nsamples
    def __getitem__(self, index):
        # randomly select a cest data
        index = random.randint(0,len(self.datadict['sat'])-1)
        kf    = self.datadict['sat']['kf'][index]
        cb    = self.datadict['sat']['cb'][index]
        csm   = self.datadict['csm'][index]

        noffs = kf.shape[1]
        # randomly select a sequence of adjacent images
        index = random.randint(0,noffs-self.nadj)
        kf = kf[:,index:index+self.nadj]            # (ncoils, nadj, height, width)
        cb = ifft2c(cb[index:index+self.nadj])      # (1, nadj, height, width)
        xf = ifft2c(kf)                             # (ncoils, nadj, height, width)
        return xf, kf, cb, csm

def main(args):
    # turn on cudnn to speed up
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True

    # construct fitting model
    model = getattr(getattr(models, args.model), args.model)
    with open(os.path.join('params',args.model+'.yaml'), 'r') as f:
        modelparams = yaml.load(f, yaml.SafeLoader)
    model = model(modelparams).cuda()
    model.load_state_dict(torch.load(args.checkpoint))

    # construct training dataset
    trainset = dataset(args.datapath, args.csmpath, modelparams['nadj'])
    trainloader = DataLoader(trainset, modelparams['finetune']['batch_size'], shuffle=True, pin_memory=True)

    # run training
    epochs = modelparams['finetune']['epochs']
    lr, momentum, weight_decay = modelparams['finetune']['lr'], modelparams['momentum'], modelparams['weight_decay']
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = PolyScheduler(optimizer, t_total=epochs)
    criterion = nn.L1Loss()
    iter = 0
    writer = SummaryWriter()
    for epoch in range(epochs):
        for xf, kf, cb, csm in trainloader:
            ku = model.undersampling(kf, mode='sense').cuda()
            xf, cb, csm = xf.cuda(), cb.cuda(), csm.cuda()
            
            xr, xcb = model(ku, csm, mode='sense')
            loss = criterion(abs(xr), abs(xf)) + modelparams['finetune']['gamma'] * criterion(abs(xcb), abs(cb))
            
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
            plt.imshow(abs(xcb)[0][0].detach().cpu())
            plt.subplot(1,2,2)
            plt.imshow(abs(cb)[0][0].detach().cpu())
            plt.show(block=False)
            plt.pause(3)
            plt.close()
        # save at every x epoch
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join('checkpoints',args.model,'finetune',f'ckp_{epoch+1}.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='reconstruction model')
    parser.add_argument('--datapath', type=str, help='abs. path of training data')
    parser.add_argument('--csmpath', type=str, help='abs. path of coil sensitivity maps')
    parser.add_argument('--checkpoint', type=str, help='abs. path of pretrained model')
    args = parser.parse_args()
    main(args)