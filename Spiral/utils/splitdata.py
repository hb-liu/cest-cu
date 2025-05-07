# This script split dataset into training and testing at given ratio, 
# results are stored in databatch folder
import os
import random
import argparse

def main(args):
    datapath = args.datapath
    ratio = args.ratio
    
    names = os.listdir(datapath)
    random.shuffle(names)

    nsamples = len(names)
    trains = names[:int(nsamples*ratio/(ratio+1))]
    tests  = names[int(nsamples*ratio/(ratio+1)):]

    with open('databatch/trains.txt','w') as f:
        for name in trains:
            f.write(name+'\n')

    with open('databatch/tests.txt','w') as f:
        for name in tests:
            f.write(name+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, help='abs. path of dataset')
    parser.add_argument('--ratio', type=float, help='ratio = #train/#test')
    args = parser.parse_args()
    main(args)