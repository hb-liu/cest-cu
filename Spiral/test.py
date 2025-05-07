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
    with open('databatch/tests.txt','r') as f:
        names = f.read().splitlines()
    
    nadj = modelparams['nadj']
    for name in names:
        data = loadmat(os.path.join(args.datapath,name),simplify_cells=True)['data']
        sat, usat, trajs, offsets = data['sat'], data['usat'], data['trajs'], data['offsets']
        sat   = np.transpose(sat,[2,0,1])      # (noffs, traj_length, nleaves)
        trajs = np.transpose(trajs,[3,0,1,2])  # (noffs, ndim, traj_length, nleaves)
        
        # fetch undersampled k-space of saturated and fullysampled image of unsaturated (m0)
        sat   = torch.from_numpy(sat)
        usat  = torch.from_numpy(usat)
        trajs = torch.from_numpy(trajs)

        # run reconstruction one by one, averaging results of the same image
        noffs, height, width = sat.shape[0], usat.shape[0], usat.shape[1]
        rec  = torch.zeros([noffs, height, width], dtype=sat.dtype)
        cnt  = torch.zeros([noffs, height, width])
        for i in range(noffs-nadj+1):
            ku = sat[i:i+nadj][None].cuda()
            tu = trajs[i:i+nadj].cuda()
            m0 = usat[None][None].cuda()
            with torch.no_grad():
                xr = model(ku,tu,m0)
            rec[i:i+nadj] += xr[0].cpu()
            cnt[i:i+nadj] += 1
        rec = rec / cnt
        rec = rec.numpy()
        rec = np.transpose(rec,[1,2,0])

        # store reconstrution results
        data = {}
        data['sat']     = rec       # reconstructed image, (height, width, noffs), complex
        data['usat']    = usat      # fullysampled, unsaturated image, (height, width), complex
        data['offsets'] = offsets   # saturation frequency offsets, m0 not included
        savemat(os.path.join('results',args.model,name), {'data': data})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='reconstruction model')
    parser.add_argument('--datapath', type=str, help='abs. path of testing data')
    parser.add_argument('--checkpoint', type=str, help='checkpoint of pretrained model')
    args = parser.parse_args()
    main(args)