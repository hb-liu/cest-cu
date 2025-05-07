import os
import yaml
import torch
import models
import argparse
import numpy as np
from utils.utils import *
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

def main(args):
    # construct fitting model
    model = getattr(getattr(models, args.model), args.model)
    with open(os.path.join('params',args.model+'.yaml'), 'r') as f:
        modelparams = yaml.load(f, yaml.SafeLoader)
    model = model(modelparams).cuda()
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()

    # run testing on databatch
    with open(args.databatch,'r') as f:
        names = f.read().splitlines()
    
    nadj = modelparams['nadj']
    for name in names:
        data = loadmat(os.path.join(args.datapath,name),simplify_cells=True)['data']
        kspace, offsets = data['kspace'], data['offsets']
        kspace = np.transpose(kspace,[2,0,1])
        sat    = kspace[offsets<100]
        usat   = np.mean(kspace[offsets>=100],0,keepdims=True)
        sat    = torch.from_numpy(sat)
        usat   = torch.from_numpy(usat)

        # run reconstruction one by one, averaging results of the same image
        rec  = torch.zeros_like(sat)
        cnt  = torch.zeros_like(sat)
        for i in range(len(sat)-nadj+1):
            ku = sat[i:i+nadj]
            ku, m0 = ku[None].cuda(), ifft2c(usat)[None].cuda()
            with torch.no_grad():
                xr = model(ku,m0)
            rec[i:i+nadj] += xr[0].cpu()
            cnt[i:i+nadj] += 1
        rec = rec / cnt
        rec = fft2c(rec).numpy()
        rec = np.transpose(rec,[1,2,0])

        # store reconstrution results
        kspace = np.zeros_like(data['kspace'])
        kspace[:,:,offsets<100] = rec
        kspace[:,:,offsets>=100] = data['kspace'][:,:,offsets>=100]
        data = {}
        data['kspace'] = kspace
        data['offsets'] = offsets
        savemat(os.path.join('results',args.model,os.path.basename(name)), {'data': data})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='reconstruction model')
    parser.add_argument('--datapath', type=str, help='abs. path of testing data')
    parser.add_argument('--checkpoint', type=str, help='checkpoint of pretrained model')
    parser.add_argument('--databatch', type=str, help='abs. path of databatch file')
    args = parser.parse_args()
    main(args)