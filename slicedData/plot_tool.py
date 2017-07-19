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
from itertools import product, combinations

def plot_2d_pc_bbox(grid, xmin, ymin, xmax, ymax, layer, typedata):
    fig = pylab.figure()
    ax = fig.add_subplot(111, aspect='equal')
    xlen=  xmax - xmin
    ylen=  ymax - ymin
    for i in xrange(len(xmin)):
        ax.add_patch(patches.Rectangle( (xmin[i],ymin[i]), xlen[i], ylen[i], fill=False, edgecolor='green' ))
        #ax.add_patch(patches.Rectangle( (ymin[i],xmin[i]), ylen[i], xlen[i], fill=False, edgecolor='green' ))
    plt.imshow(grid)
    plt.draw()
    plt.title('layer:'+'%d'%(layer)+ 'type: %s'%(typedata))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()
    plt.show(block=False)


def plot_3d(pc, xmin, xmax, ymin, ymax, zmin, zmax):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    '''
    color = ['r','g','b','r','g','b','r','g','b','r','g','b','r','g','b','r','g','b','r','g','b','r','g','b','r']
    for num in xrange(len(layer)):
        xs = layer[num][0][range(0,len(layer[num][0]),10)]
        ys = layer[num][1][range(0,len(layer[num][1]),10)]
        zs = layer[num][2][range(0,len(layer[num][2]),10)]
        xb = np.concatenate((xmin,xmax))
        yb = np.concatenate((zmin,xmax))
        zb = np.concatenate((zmin,zmax))
        ax.scatter(xs, ys, zs, c=color[num], marker='o')
        #ax.scatter(xb, yb, zb, c='y', marker='^')
        ax.scatter(xmin, ymin, zmin, c='y', marker='^')
        ax.scatter(xmax, ymax, zmax, c='k', marker='^')
        #plt.setp(ax.scatter(0, 0, 0), color='yellow')
    '''
    color = ['r']
    xs = pc[0][range(0,len(pc[0]),50)]
    ys = pc[1][range(0,len(pc[1]),50)]
    zs = pc[2][range(0,len(pc[2]),50)]
    ax.scatter(xs, ys, zs, c=color, marker='o')
    for kk in xrange(len(xmin)):
        rx = [xmin[kk], xmax[kk]]
        ry = np.nanmax(ys) - [ymax[kk], ymin[kk]]
        rz = [zmin[kk], zmax[kk]]
        for ss, ee in combinations(np.array(list(product(rx, ry, rz))), 2):
            edge = np.sum(np.abs(ss-ee))
            if edge == rx[1]-rx[0] or edge == ry[1]-ry[0] or edge == rz[1]-rz[0]:
                ax.plot3D(*zip(ss, ee), color="b")
    #ax.scatter(xmin, ymin, zmin, c='y', marker='^')
    #ax.scatter(xmax, ymax, zmax, c='k', marker='^')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

