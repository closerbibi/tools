import os 
import numpy as np
import pdb

path = '../data/hhabv/projecting/'

pixel_count = 0
sum_val = np.zeros((3))
for fname in os.listdir(path):
    tmp = np.load(path+fname)
    sum_val += np.sum(np.sum(tmp,axis=2),axis=1)
    pixel_count += tmp.shape[1]*tmp.shape[2]


print sum_val/pixel_count
    

