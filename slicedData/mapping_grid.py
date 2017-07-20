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
from occlusion import *
from plot_tool import *

def hhagrid(lpc_id, hha, imagenum, grid, idxx, idxy):
    # pc: (x, z, -y), z is depth
    ## sorting array by x
    lpc_id = lpc_id[:,~np.isnan(lpc_id[0])]
    lpc_trans = np.transpose(lpc_id,(1,0))
    lpc_x_sort = lpc_trans[lpc_trans[:, 0].argsort()]
    lpc_x_sort = np.transpose(lpc_x_sort,(1,0))

    for ix in idxx:
        x_loc = np.where(lpc_x_sort[0].astype(int)==ix)
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            xy_eligible = np.where(lpc_x_sort[1][x_loc].astype(int)==iy)[0]
            # channel 0: angle
            # channel 1: height
            # channel 2: disparity
            if len(xy_eligible) > 0:
                # then find the position of max z to complete the map, bv can only see the highest point
                z_lst = lpc_x_sort[2,x_loc][0,xy_eligible]
                z_lst_argsort = (np.argsort(z_lst))[::-1]# sort=small>large, use[::-1] to large>small~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                for i in range(len(z_lst_argsort)):
                    original_idx = xy_eligible[z_lst_argsort[i]]
                    #original_idx1 = xy_eligible[np.argmax(z_lst)]
                    location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                    #location1 = lpc_x_sort[3,x_loc][0, original_idx1].astype(int)
                    #if hha[location,2] < 190: # pruning roof base on angle
                    if hha[location,1] < 170: # pruning roof base on height
                        grid[0][rviy][ix] = hha[location,2] # angle
                        grid[1][rviy][ix] = hha[location,2] # height
                        grid[2][rviy][ix] = hha[location,2] # disparity
                        ######### All angle ############
                        break
                    else:
                        continue
            else:
                grid[0][rviy][ix] = 0
                grid[1][rviy][ix] = 0 #np.nanmin(height)
                grid[2][rviy][ix] = 0
    #plt.imshow(grid[0]);plt.colorbar();plt.show()
    return grid

def constructing_grid_pj_prune_roof(max_idxy,idxx,idxy,grid,lpc_id,img_idx,occu_map):

    ### color image or hha image, both are (2,1,0)
    hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
    hha = cv2.imread(hha_name)
    bgr_name = '/home/closerbibi/workspace/data/NYUimg_only/NYU%04d.jpg'%(int(img_idx))
    bgr = cv2.imread(bgr_name)
    #gray_image = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2GRAY)
    angle = hha[:,:,0] # hha: angle, blue
    height = hha[:,:,1] # hha: height, green
    disparity = hha[:,:,2] # hha: disparity, red
    # choose img type
    img = hha # !!!!!!!!!!!!!!!

    ## sorting array by x
    lpc_id = lpc_id[:,~np.isnan(lpc_id[0])]
    lpc_trans = np.transpose(lpc_id,(1,0))
    lpc_x_sort = lpc_trans[lpc_trans[:, 0].argsort()]
    lpc_x_sort = np.transpose(lpc_x_sort,(1,0))

    hh = height.shape[0]; ww = height.shape[1];
    for ix in idxx:
        x_loc = np.where(lpc_x_sort[0].astype(int)==ix)
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            xy_eligible = np.where(lpc_x_sort[1][x_loc].astype(int)==iy)[0]
            # channel 0: angle
            # channel 1: height
            # channel 2: disparity
            if len(xy_eligible) > 0:
                # then find the position of max z to complete the map, bv can only see the highest point
                z_lst = lpc_x_sort[2,x_loc][0,xy_eligible]
                z_lst_argsort = (np.argsort(z_lst))[::-1]# sort=small>large, use[::-1] to large>small~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                for i in range(len(z_lst_argsort)):
                    original_idx = xy_eligible[z_lst_argsort[i]]
                    #original_idx1 = xy_eligible[np.argmax(z_lst)]
                    location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                    #location1 = lpc_x_sort[3,x_loc][0, original_idx1].astype(int)
                    wmap = int(np.floor(location/hh))
                    hmap = (location%hh)-1
                    if angle[hmap,wmap] < 190: # pruning roof base on angle
                        grid[0][rviy][ix] = img[:,:,0][hmap,wmap]
                        grid[1][rviy][ix] = img[:,:,1][hmap,wmap]
                        grid[2][rviy][ix] = img[:,:,2][hmap,wmap]
                        occu_map[rviy][ix] = 0
                        break
                    else:
                        continue
            else:
                grid[0][rviy][ix] = 0
                grid[1][rviy][ix] = 0 #np.nanmin(height)
                grid[2][rviy][ix] = 0
    #grid[2] = occu_map
    return grid


def constructing_grid_pj(max_idxy,idxx,idxy,grid,lpc_id,img_idx):

    ### color image or hha image, both are (2,1,0)
    #hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
    #img = cv2.imread(hha_name)
    rgb_name = '/home/closerbibi/workspace/data/NYUimg_only/NYU%04d.jpg'%(int(img_idx))
    img = cv2.imread(rgb_name)
    disparity = img[:,:,2] # hha: disparity
    height = img[:,:,1] # hha: height
    angle = img[:,:,0] # hha: angle

    ## sorting array by x
    lpc_id = lpc_id[:,~np.isnan(lpc_id[0])]
    lpc_trans = np.transpose(lpc_id,(1,0))
    lpc_x_sort = lpc_trans[lpc_trans[:, 0].argsort()]
    lpc_x_sort = np.transpose(lpc_x_sort,(1,0))

    for ix in idxx:
        x_loc = np.where(lpc_x_sort[0].astype(int)==ix)
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            xy_eligible = np.where(lpc_x_sort[1][x_loc].astype(int)==iy)[0]
            # channel 0: disparity
            # channel 1: height
            # channel 2: angle
            if len(xy_eligible) > 0:
                # then find the position of max z to complete the map, bv can only see the highest point
                original_idx = xy_eligible[np.argmax(lpc_x_sort[2,x_loc][0,xy_eligible])]
                location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                hh = height.shape[0]; ww = height.shape[1];
                wmap = int(np.floor(location/hh))
                hmap = (location%hh)-1
                grid[0][rviy][ix] = angle[hmap,wmap]
                grid[1][rviy][ix] = height[hmap,wmap]
                grid[2][rviy][ix] = disparity[hmap,wmap]
            else:
                grid[0][rviy][ix] = 0
                grid[1][rviy][ix] = 0 #np.nanmin(height)
                grid[2][rviy][ix] = 0

    return grid


def constructing_grid_ly(max_idxy,idxx,idxy,grid,large_layerpc,cur_floor,pc,angle):

    ## sorting array by x
    lpc_id = large_layerpc[:,~np.isnan(large_layerpc[0])]
    lpc_trans = np.transpose(lpc_id,(1,0))
    lpc_x_sort = lpc_trans[lpc_trans[:, 0].argsort()]
    lpc_x_sort = np.transpose(lpc_x_sort,(1,0))

    for ix in idxx:
        x_loc = np.where(lpc_x_sort[0].astype(int)==ix)
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            xy_eligible = np.where(lpc_x_sort[1][x_loc].astype(int)==iy)[0]

            if large_layerpc.shape[1] == 0 or len(xy_eligible) == 0: #large_layerpc[2][location].shape[0] == 0
                grid[0][rviy][ix] = 0
            else:
                # then find the position of max z to complete the map, bv can only see the highest point
                original_idx = xy_eligible[np.argmax(lpc_x_sort[2,x_loc][0,xy_eligible])]
                location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                hh = angle.shape[0]; ww = angle.shape[1];
                wmap = int(np.floor(location/hh))
                hmap = (location%hh)-1
                grid[0][rviy][ix] = angle[hmap, wmap]
    return grid

