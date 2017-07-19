import scipy.io as sio
import os
import numpy as np
import h5py
import pdb
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import pylab
import matplotlib.patches as patches
#from joblib import Parallel, delayed
from multiprocessing import Pool
import itertools
import rankpooling as rp
import cv2
from functools import partial
from PIL import Image
from numpy.linalg import norm
import numba as nb # vectorize & jit
import math
import linalg_3 as lin3


@nb.jit(nopython=True)
def square(x):
    return x ** 2

@nb.jit(nopython=True)
def numbal2(x, y): 
    return math.sqrt(square(x) + square(y))


#@nb.vectorize(["float64(float64,float64,float64)"], target='cuda')
def voxel_dis(current_pc, cam_center, location):
    # location : [i,jj,k]
    #p1p2 = cam_center - location;
    #normp1p2 = lin3.norm(p1p2)
    #dis = lin3.norm(lin3.cross(p1p2, current_pc - location))/normp1p2
    # line: p1, p2; point: p3i
    # d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    #dis = norm(np.cross(p1p2, current_pc - location))/normp1p2
    print 'not implement'

def voxel_occlu(voxel_world, cam_center, idxx, idxy, idxz, imagenum):
    # !! cam_center: x,y,z <--> occlu: y,x,z(for visualization)
    occlu = np.zeros((len(idxy),len(idxx),len(idxz)))
    thre = 5 
    diss = []
    t1 = time.time()
    for i in idxx:
        for jj in idxy:
            for k in idxz:
                count = 0 
                p1p2 = cam_center-[i,jj,k]
                normp1p2 = norm(p1p2)
                for num in xrange(voxel_world.shape[1]):
                    # line: p1, p2; point: p3
                    # d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
                    dis = norm(np.cross(p1p2, voxel_world[:3,num] - np.array([i,jj,k])))/normp1p2
                    #diss += [dis]
                    if dis < thre:
                        count += 1
                    if count >= 1:
                        rev_y = len(idxy)-1 - jj
                        occlu[rev_y,i,k] = 100 # y, x, z
                        continue

    print time.time()-t1
    tmp = 'tmp/occlu_%06d.npy'%(int(imagenum))
    np.save(tmp, occlu)
    pdb.set_trace()

def z_buffer(voxel_world, idxx, idxy, idxz):
    # create occlusion map
    occu = np.zeros((len(idxy),len(idxx),len(idxz)))

    ## sorting array by x
    voxel_world = voxel_world[:,~np.isnan(voxel_world[0])]
    voxel_trans = np.transpose(voxel_world,(1,0))
    voxel_x_sort = voxel_trans[voxel_trans[:, 0].argsort()]
    voxel_x_sort = np.transpose(voxel_x_sort,(1,0))
    # looping over each x-z pixel
    for ix in idxx:
        x_loc = np.where(voxel_x_sort[0].astype(int)==ix)
        for iz in idxz:
            # location of occupied pixel
            xy_eligible = np.where(voxel_x_sort[2][x_loc].astype(int)==iz)[0]
            if len(xy_eligible) > 0:
                # find occlusion starting point
                starty = int(np.nanmin(voxel_x_sort[1][x_loc][xy_eligible]))
                rev_starty = len(idxy)-1 - starty
                occu[0:rev_starty,ix,iz] = 100 

    occu_map = np.max(occu, axis=2)
    return occu_map

