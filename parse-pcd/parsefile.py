import os
import numpy as np
import pdb
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='get header line length')
    parser.add_argument('--hl', dest='hl', 
                        help='get header line length',
                        default=11, type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    pcdpath = '/home/closerbibi/workspace/data/upsample/'
    flist = sorted(os.listdir(pcdpath))
    for num in flist:
        if 'image' in num:
            if not os.path.exists('output1000k/{}_pc.npy'.format(num)):
                with open(pcdpath+'{}/annotation_pc-upsample_r3_100.pcd'.format(num),'r') as f:
                    freadlines = f.readlines()
                    pc = np.zeros((len(freadlines)-args.hl,6)) # remove header line of pcd file
                    t1 = time.time()
                    for idx, line in enumerate(freadlines):
                        if idx < args.hl:
                            continue
                        data = line.split(' ')
                        pc[idx-args.hl][0] = float(data[0]) 
                        pc[idx-args.hl][1] = float(data[1])
                        pc[idx-args.hl][2] = float(data[2])
                        pc[idx-args.hl][3] = float(data[3])
                        pc[idx-args.hl][4] = float(data[4])
                        pc[idx-args.hl][5] = float(data[5])
                np.save('output1000k/{}_pc.npy'.format(num), pc)
                print "{} done, time: {:f}".format(num, time.time()-t1)
