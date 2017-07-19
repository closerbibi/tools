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
from occlusion import *
from plot_tool import *
from mapping_grid import *

def rescale_upsampling_pc(pc):
    pcxmin = np.nanmin(pc[0]);pcxmax = np.nanmax(pc[0])
    pcymin = np.nanmin(pc[1]);pcymax = np.nanmax(pc[1])
    pczmin = np.nanmin(pc[2]);pczmax = np.nanmax(pc[2])
    pcx_shift = pc[0]-pcxmin
    pcy_shift = pc[1]-pcymin
    pcz_shift = pc[2]-pczmin
    largex = np.floor(pcx_shift)
    largey = np.floor(pcy_shift)
    largez = np.floor(pcz_shift)
    lpc_id = np.vstack((largex,largey,largez,range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)

    return lpc_id, idxx, idxy

def rescale_pc_200(pc):
    pcx_shift = pc[0]-np.nanmin(pc[0])
    pcy_shift = pc[1]-np.nanmin(pc[1])                                                                                                                                                                                                                                                                                                                                                                                                                                         
    pcz_shift = pc[2]-np.nanmin(pc[2])
    largex = np.floor(pcx_shift*200)
    largey = np.floor(pcy_shift*200)
    largez = np.floor(pcz_shift*200)
    lpc_id = np.vstack((largex,largey,pc[2],range(len(pc[2]))))
    # new voxelized z point cloud
    voxel_world = np.vstack((largex,largey,largez,range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)
    idxz = range(int(np.nanmin(voxel_world[2])),int(np.nanmax(voxel_world[2])+1),1)
    cam_center = np.array([np.nanmin(pc[0]), np.nanmin(pc[1]), np.nanmin(pc[2])])
    cam_center = np.floor(np.absolute(cam_center*200))*[1,-1,1]
    return idxx,idxy,idxz,lpc_id,cam_center, voxel_world


def rescale_box_200(xmin, ymin, xmax, ymax, zmin, zmax, pc, idxx, idxy, clss, imagenum):
    xmin = xmin - np.nanmin(pc[0]);
    xmax = xmax - np.nanmin(pc[0]);
    ymin = ymin - np.nanmin(pc[1]);
    ymax = ymax - np.nanmin(pc[1]);
    xmin = np.floor(xmin*200);
    xmax = np.floor(xmax*200);
    ymin = np.floor(ymin*200);
    ymax = np.floor(ymax*200); #zmin = np.floor(zmin*100); zmax = np.floor(zmax*100);
    evil_list = []
    for ii in xrange(len(xmin)):
         if xmin[ii] > max(idxx):
            evil_list.extend([ii])
    for ii in xrange(len(xmin)):
         if ymin[ii] > max(idxy):
            evil_list.extend([ii])
    for ii in xrange(len(xmin)):
         if ymax[ii] < 0:
            evil_list.extend([ii])
    for ii in xrange(len(xmin)):
         if xmax[ii] < 0:
            evil_list.extend([ii])

    if any(np.nanmax(idxy) < a for a in ymin) or any(0 > a for a in ymax) or  any(np.nanmax(idxx) < a for a in xmin) or  any(0 > a for a in xmax):
        go_away = np.unique(evil_list)
        xmin = np.delete(xmin, go_away)
        ymin = np.delete(ymin, go_away)
        xmax = np.delete(xmax, go_away)
        ymax = np.delete(ymax, go_away)
        zmin = np.delete(zmin, go_away)
        zmax = np.delete(zmax, go_away)
        clss = np.delete(clss, go_away)
    tmp_ymax = np.nanmax(idxy) - ymin;
    ymin = np.nanmax(idxy) - ymax;
    ymax = tmp_ymax
    for kk in xrange(len(xmin)):
        if xmin[kk] <= 0:
            xmin[kk] = 1
        if ymin[kk] <= 0:
            ymin[kk] = 1
        if xmax[kk] >= np.nanmax(idxx):
            xmax[kk] = np.nanmax(idxx)-1
        if ymax[kk] >= np.nanmax(idxy):
            ymax[kk] = np.nanmax(idxy)-1
    return xmin, ymin, xmax, ymax, zmin, zmax, clss


def rescale_pc(pc):
    pcx_shift = pc[0]-np.nanmin(pc[0])
    pcy_shift = pc[1]-np.nanmin(pc[1])                                                                                                                                                                                                                                                                                                                                                                                                                                         
    pcz_shift = pc[2]-np.nanmin(pc[2])
    largex = np.floor(pcx_shift*100)
    largey = np.floor(pcy_shift*100)
    largez = np.floor(pcz_shift*100)
    lpc_id = np.vstack((largex,largey,pc[2],range(len(pc[2]))))
    # new voxelized z point cloud
    voxel_world = np.vstack((largex,largey,largez,range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)
    idxz = range(int(np.nanmin(voxel_world[2])),int(np.nanmax(voxel_world[2])+1),1)
    cam_center = np.array([np.nanmin(pc[0]), np.nanmin(pc[1]), np.nanmin(pc[2])])
    cam_center = np.floor(np.absolute(cam_center*100))*[1,-1,1]
    return idxx,idxy,idxz,lpc_id,cam_center, voxel_world


def rescale_box(xmin, ymin, xmax, ymax, zmin, zmax, pc, idxx, idxy, clss, imagenum):
    xmin = xmin - np.nanmin(pc[0]);xmax = xmax - np.nanmin(pc[0]);
    ymin = ymin - np.nanmin(pc[1]);ymax = ymax - np.nanmin(pc[1]);
    zmin = zmin - np.nanmin(pc[2]);zmax = zmax - np.nanmin(pc[2]);
    
    xmin = np.floor(xmin*100);xmax = np.floor(xmax*100);
    ymin = np.floor(ymin*100);ymax = np.floor(ymax*100); #
    zmin = np.floor(zmin*100); zmax = np.floor(zmax*100);
    evil_list = []
    for ii in xrange(len(xmin)):
         if xmin[ii] > max(idxx):
            evil_list.extend([ii])
    for ii in xrange(len(xmin)):
         if ymin[ii] > max(idxy):
            evil_list.extend([ii])
    for ii in xrange(len(xmin)):
         if ymax[ii] < 0:
            evil_list.extend([ii])
    for ii in xrange(len(xmin)):
         if xmax[ii] < 0:
            evil_list.extend([ii])

    if any(np.nanmax(idxy) < a for a in ymin) or any(0 > a for a in ymax) or  any(np.nanmax(idxx) < a for a in xmin) or  any(0 > a for a in xmax):
        go_away = np.unique(evil_list)
        xmin = np.delete(xmin, go_away)
        ymin = np.delete(ymin, go_away)
        xmax = np.delete(xmax, go_away)
        ymax = np.delete(ymax, go_away)
        zmin = np.delete(zmin, go_away)
        zmax = np.delete(zmax, go_away)
        clss = np.delete(clss, go_away)
    tmp_ymax = np.nanmax(idxy) - ymin;
    ymin = np.nanmax(idxy) - ymax;
    ymax = tmp_ymax
    for kk in xrange(len(xmin)):
        if xmin[kk] <= 0:
            xmin[kk] = 1
        if ymin[kk] <= 0:
            ymin[kk] = 1
        if xmax[kk] >= np.nanmax(idxx):
            xmax[kk] = np.nanmax(idxx)-1
        if ymax[kk] >= np.nanmax(idxy):
            ymax[kk] = np.nanmax(idxy)-1
    return xmin, ymin, xmax, ymax, zmin, zmax, clss

