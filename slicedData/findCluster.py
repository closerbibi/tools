# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:36:15 2017

@author: closerbibi
"""

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
from sklearn.cluster import KMeans



def rescale_pc(pc):
    pcx_shift = pc[0]-np.nanmin(pc[0])
    pcy_shift = pc[1]-np.nanmin(pc[1])
    largex = np.floor(pcx_shift*100)
    largey = np.floor(pcy_shift*100)
    lpc_id = np.vstack((largex,largey,pc[2],range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)
    return idxx,idxy,lpc_id


def rescale_box(xmin, ymin, xmax, ymax, zmin, zmax, pc, idxx, idxy, clss, imagenum):
    xmin = xmin - np.nanmin(pc[0]); 
    xmax = xmax - np.nanmin(pc[0]); 
    ymin = ymin - np.nanmin(pc[1]); 
    ymax = ymax - np.nanmin(pc[1]);
    xmin = np.floor(xmin*100); 
    xmax = np.floor(xmax*100); 
    ymin = np.floor(ymin*100); 
    ymax = np.floor(ymax*100); #zmin = np.floor(zmin*100); zmax = np.floor(zmax*100);
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
        #print 'xmin: %s, picxmax: %s, ymin: %s, picymax: %s' %(xmin, max(idxx), ymin, max(idxy))
        #print 'xmax: %s, ymax: %s, 0'% (xmax, ymax)
        #print evil_list
        #print imagenum
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




if __name__ == '__main__':

    #data = h5py.File('/media/disk2/3D/understanding/rankpooling/rgbd_data/nyu_v2_labeled/nyu_depth_v2_labeled.mat')

    #calibration parameters from author
    #K = np.array([[5.7616540758591043e+02,0,3.2442516903961865e+02],[0,5.7375619782082447e+02,2.3584766381177013e+02],[0,0,1]]) 

    count=0
    count_box=0
    count_mat=0

    start_time = time.time()
    numbox = []
    newidx = np.array([89,122,146,149,153,159,254,449,725,755,790,936,1036,1038,1273,1427])
    dict_box = {k: 0 for k in xrange(25)}
    #fid_nobox = open('/home/closerbibi/workspace/pytorch/faster-rcnn_19classes/data/DIRE/nobox_image.txt','w')
    fid_nobox = open('all_cls.txt','w')
    wh_noclass = np.array([])
    for imagenum in xrange(1,1450):
        #if np.where(np.array([88, 179, 368, 390, 650])==imagenum)[0].size != 0:
        #    continue
        # bed=157, chair=5, table=19, sofa=83, toilet=124
        try:
            box_pc = sio.loadmat('alignData_with_nan_19_classes/image%04d/annotation_pc.mat' % (imagenum)); # pc generate by bin.....SUN3Dtoolbox/generate_pc_3dBox.m
            count_mat += 1
        except:
            continue
        pc = box_pc['points3d']; 
        pc = np.swapaxes(pc, 0, 1)

        #if np.where(clss=='chair')[0].size == 0:
        #    continue
        
        print 'now at image: %d' % (imagenum)
        ###print 'count: %d' %(count)
        
        

        # rescale pc
        idxx, idxy, lpc_id = rescale_pc(pc)

        # change bbox to y,x,z
        try:
            clss=box_pc['clss'][0]
            ymin = box_pc['ymin'][0]; ymax=box_pc['ymax'][0]; xmin=box_pc['xmin'][0]; xmax=box_pc['xmax'][0]; zmin=box_pc['zmin'][0]; zmax=box_pc['zmax'][0];
            # rescale box to fit image size
            xmin, ymin, xmax, ymax, zmin, zmax, clss = rescale_box(xmin, ymin, xmax, ymax, zmin, zmax, pc, idxx, idxy, clss, imagenum)
            clss[0]
        except:
            #reduct dimension to 2D
            fid_nobox.write('%d\n'%(imagenum))
            st1 = time.time()
            #print time.time()-st1
            # write bbox file
            #fid = open('../../data/label_19/picture_%06d.txt'%(imagenum),'w')
            #fid.write('')
            #fid.close()
            ### making no box list
            count+=1
            continue

        ww = xmax - xmin
        hh = ymax - ymin

        if imagenum ==1:
            wh_noclass = np.vstack((ww,hh)).transpose(1,0)
        else:
            wh_noclass = np.vstack((wh_noclass,np.vstack((ww,hh)).transpose(1,0)))
        '''
        for k in xrange(len(clss)):
            try:
                if (clss[k][0]) == 'nightstand':
                    
                else:
            except:
                print 'Ooooooops'
        '''
        #numbox.extend([5000+imagenum])
        count_box+=1
    # find center of cluster
    wh_noclass = wh_noclass[np.where(~np.isnan(wh_noclass[:,0]))[0].tolist(),:]
    pdb.set_trace()
    km = KMeans(n_clusters=11, random_state=0).fit(wh_noclass)
    print km.labels_
    print km.cluster_centers_

