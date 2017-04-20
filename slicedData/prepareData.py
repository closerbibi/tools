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


'''
def constructing_grid_multi(max_idxy,ix,iy,grid,large_layerpc):
    rviy = max_idxy - iy
    # compute density: channel 0
    location = (large_layerpc[0]==ix)*(large_layerpc[1]==iy)
    grid[0][rviy][ix] = np.sum((large_layerpc[0]==ix)*(large_layerpc[1]==iy))
    # find max: channel 1
    if grid[0][rviy][ix] == 0:   # if empty, giving lowest value of current layer
        grid[1][rviy][ix] = np.nanmin(large_layerpc[2])
    else:
        grid[1][rviy][ix] = np.nanmax(large_layerpc[2][location])
    return grid

def func_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return constructing_grid(*a_b)

def constructing_grid(max_idxy,ix,iy,grid,large_layerpc):
    #idxx = inparam[0]; idxy = inparam[1]; grid = inparam[2]; large_layerpc = inparam[3]
    #for ix in idxx:
    #    for iy in idxy:
    rviy = max_idxy - iy
    # compute density: channel 0
    location = (large_layerpc[0]==ix)*(large_layerpc[1]==iy)
    grid[0][rviy][ix] = np.sum((large_layerpc[0]==ix)*(large_layerpc[1]==iy))
    # find max: channel 1
    if grid[0][rviy][ix] == 0:   # if empty, giving lowest value of current layer
        grid[1][rviy][ix] = np.nanmin(large_layerpc[2])
    else:
        grid[1][rviy][ix] = np.nanmax(large_layerpc[2][location])
    return grid
'''
def normalize_data():
    print 'iamempty'


def compute_mean():
    print "Hello, I am empty"

def rescale_pc(pc):
    pcx_shift = pc[0]-np.nanmin(pc[0])
    pcy_shift = pc[1]-np.nanmin(pc[1])
    largex = np.floor(pcx_shift*100)
    largey = np.floor(pcy_shift*100)
    lpc_id = np.vstack((largex,largey,pc[2],range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)
    return idxx,idxy,lpc_id


def rescale_box(xmin, ymin, xmax, ymax, pc, idxx, idxy):
    xmin = xmin - np.nanmin(pc[0]); 
    xmax = xmax - np.nanmin(pc[0]); 
    ymin = ymin - np.nanmin(pc[1]); 
    ymax = ymax - np.nanmin(pc[1]);
    xmin = np.floor(xmin*100); 
    xmax = np.floor(xmax*100); 
    ymin = np.floor(ymin*100); 
    ymax = np.floor(ymax*100); #zmin = np.floor(zmin*100); zmax = np.floor(zmax*100);
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
    return xmin, ymin, xmax, ymax

def constructing_grid_pj(max_idxy,idxx,idxy,grid,pc,lpc_id,img_idx):

    ### color image or hha image, both are (2,1,0)
    #hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
    #img = cv2.imread(hha_name)
    color_name = '/home/closerbibi/workspace/data/NYUimg_only/NYU%04d.jpg'%(int(img_idx))
    img = cv2.imread(color_name)
    disparity = img[:,:,2] # hha: disparity
    height = img[:,:,1] # hha: height
    angle = img[:,:,0] # hha: angle

    ## sorting array by x
    lpc_trans = np.transpose(lpc_id,(1,0))
    lpc_x_sort = lpc_trans[lpc_trans[:, 0].argsort()]
    lpc_x_sort = np.transpose(lpc_x_sort,(1,0))

    for ix in idxx:
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            pdb.set_trace()
            xy_eligible = (lpc_id[0].astype(int)==ix)*(lpc_id[1].astype(int)==iy) # "*" means binary operator "and" 
            xy_eligible = np.where(xy_eligible==True)[0]
            # channel 0: disparity
            # channel 1: height
            # channel 2: angle
            if len(xy_eligible) > 0:
                pdb.set_trace()
                # then find the max z position to complete the map, bv can only see the highest point
                location = xy_eligible[np.argmax(pc[2][xy_eligible])]
                hh = height.shape[0]; ww = height.shape[1];
                wmap = int(np.floor(location/hh))
                hmap = (location%hh)-1
                grid[0][rviy][ix] = disparity[hmap,wmap]
                grid[1][rviy][ix] = height[hmap,wmap]
                grid[2][rviy][ix] = angle[hmap,wmap]
            else:
                grid[0][rviy][ix] = 0
                grid[1][rviy][ix] = 0 #np.nanmin(height)
                grid[2][rviy][ix] = 0

            #if largexy.shape[1] == 0:
            #    grid[1][rviy][ix] = cur_floor #np.min(pc[2])
            #    grid[2][rviy][ix] = np.nanmin(pc[2])
            #else:
            #    layermax = np.nanmax(largexy[2][location])
            #    grid[1][rviy][ix] = layermax
            #    grid[2][rviy][ix] = layermax

    pdb.set_trace()
    return grid


def constructing_grid(max_idxy,idxx,idxy,grid,large_layerpc,cur_floor,pc):

    for ix in idxx:
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # compute density: channel 0
            location = (large_layerpc[0]==ix)*(large_layerpc[1]==iy)
            grid[0][rviy][ix] = np.sum((large_layerpc[0]==ix)*(large_layerpc[1]==iy))

            # find max: channel 1
            if large_layerpc.shape[1] == 0:
                grid[1][rviy][ix] = cur_floor #np.min(pc[2])
                grid[2][rviy][ix] = np.nanmin(pc[2])
            elif large_layerpc[2][location].shape[0]==0:   # if empty, giving lowest value among all pc
                grid[1][rviy][ix] = np.nanmin(large_layerpc[2]) #np.nanmin(pc[2])
                grid[2][rviy][ix] = np.nanmin(pc[2])
            else:
                layermax = np.nanmax(large_layerpc[2][location])
                grid[1][rviy][ix] = layermax
                grid[2][rviy][ix] = layermax
    return grid

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


def plot_3d(layer, xmin, xmax, ymin, ymax, zmin, zmax, layers):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def to2D(pc, imagenum, xmin, ymin, xmax, ymax, zmin, zmax, idxx, idxy, lpc_id, method, img_idx):
    points_height = pc[2,:]
    ceiling = np.nanmax(points_height)
    floor = np.nanmin(points_height)

    # seperate to multiple layers
    layers = 100 # 25
    l_step = (ceiling - floor)/layers
    layer = {}

    data_grid = np.meshgrid(idxx,idxy)
    s_den = {}
    s_maxcur = {}
    s_maxgrd = {}
    sw=0

    if method == 'projecting':
        grid_file = '../../data/rgbbv/projecting/picture_%06d.npy'%(imagenum)
        grid = np.zeros((3,len(idxy),len(idxx)))
        max_idxy = np.nanmax(idxy)
        if os.path.exists(grid_file):
            grid = constructing_grid_pj(max_idxy,idxx,idxy,grid,pc,lpc_id,img_idx)
            print "grid file done."
        else:
            grid = constructing_grid_pj(max_idxy,idxx,idxy,grid,pc,lpc_id,img_idx)
            np.save(grid_file,grid)
        # plot 2d to show box and data
        #for kk in xrange(3):
        #plot_2d_pc_bbox(grid[kk], xmin, ymin, xmax, ymax, kk, 'proj')
        #plot_2d_pc_bbox(scene_den[kk-4], xmin, ymin, xmax, ymax, kk, 'den')

    if method == 'rankpooling':

        grid_file = '../../data/hhabv/rankpooling/picture_%06d.npy'%(imagenum)
        grid = np.zeros((3,len(idxy),len(idxx)))
        max_idxy = np.nanmax(idxy)
        if os.path.exists(grid_file):
            print "grid file done."
        else:
            # find index in the range
            for ilayer in xrange(layers):#xrange(layers):
                cur_ceiling = floor + l_step*(ilayer+1)
                cur_floor = floor + l_step*ilayer

                idx = np.where((pc[2,:] > cur_floor) * (pc[2,:] < cur_ceiling))
                layer[ilayer] = pc[:,idx[0]]

                # show 3d data and bbox corner
                #plot_3d(layer, xmin, ymin, xmax, ymax, zmin, zmax, layers) 

                # making grid
                layerpcx_shift = layer[ilayer][0]-np.nanmin(pc[0])
                layerpcy_shift = layer[ilayer][1]-np.nanmin(pc[1])
                layerlargex = np.floor(layerpcx_shift*100)
                layerlargey = np.floor(layerpcy_shift*100)
                large_layerpc = np.vstack((layerlargex,layerlargey,layer[ilayer][2]))
                grid = np.zeros((3,len(idxy),len(idxx)))
                max_idxy = np.nanmax(idxy)

                grid = constructing_grid(max_idxy,idxx,idxy,grid,large_layerpc,cur_floor,pc)
                s_den[ilayer] = grid

                arrays_den = [s_den[x] for x in layer.keys()]
                scene_den = np.stack(arrays_den, axis=0)

                np.save(grid_file,scene_den)
                # plot 2d to show box and data
                '''
                for kk in layer.keys():
                    plot_2d_pc_bbox(scene_den[kk-4], xmin, ymin, xmax, ymax, kk, 'den')
                    plot_2d_pc_bbox(scene_maxcur[kk-4], xmin, ymin, xmax, ymax, kk, 'maxcur')
                    plot_2d_pc_bbox(scene_maxgrd[kk-4], xmin, ymin, xmax, ymax, kk, 'maxgrd')
                '''


    if method == 'layered':
        # find index in the range
        # according to the analyse, picking layers who are most polular
        for ilayer in xrange(3,9):#xrange(layers):
            cur_ceiling = floor + l_step*(ilayer+1)
            cur_floor = floor + l_step*ilayer
            '''
            # trim layer without target class (chair only first)
            # change to more accurate layer finding
            if (cur_ceiling > zmax).all() or (cur_floor < zmin).all():
                continue
            '''
            idx = np.where((pc[2,:] > cur_floor) * (pc[2,:] < cur_ceiling))
            layer[ilayer] = pc[:,idx[0]]

            # show 3d data and bbox corner
            #plot_3d(layer, xmin, ymin, xmax, ymax, zmin, zmax, layers) 
            # making grid
            layerpcx_shift = layer[ilayer][0]-np.nanmin(pc[0])
            layerpcy_shift = layer[ilayer][1]-np.nanmin(pc[1])
            layerlargex = np.floor(layerpcx_shift*100)
            layerlargey = np.floor(layerpcy_shift*100)
            large_layerpc = np.vstack((layerlargex,layerlargey,layer[ilayer][2]))
            grid = np.zeros((3,len(idxy),len(idxx)))
            # using parallel computing to boost up
            #grid = Parallel(n_jobs=num_cores)(delayed(constructing_grid)(idxx,idxy,grid,large_layerpc))
            #pool = Pool()
            #inparam = {}
            #inparam['idxx'] = idxx; inparam['idxy'] = idxy; inparam['grid'] = grid; inparam['large_layerpc'] = large_layerpc;
            #inparam = [idxx,idxy,grid,large_layerpc]
            #grid = pool.map(constructing_grid, inparam)
            #constructing_grid(idxx,idxy,grid,large_layerpc)

            # max_idxy,ix,iy,grid,large_layerpc
            max_idxy = np.nanmax(idxy)
            '''
            for ix in idxx:
                pool.map(func_star, itertools.izip(itertools.repeat(max_idxy),itertools.repeat(ix),idxy,itertools.repeat(grid),itertools.repeat(large_layerpc)))

            '''

            grid = constructing_grid(max_idxy,idxx,idxy,grid,large_layerpc,cur_floor,pc)
            s_den[ilayer] = grid[0]
            s_maxcur[ilayer] = grid[1]
            s_maxgrd[ilayer] = grid[2]

            arrays_den = [s_den[x] for x in layer.keys()]
            arrays_maxcur = [s_maxcur[y] for y in layer.keys()]
            arrays_maxgrd = [s_maxgrd[y] for y in layer.keys()]
            scene_den = np.stack(arrays_den, axis=0)
            scene_maxcur = np.stack(arrays_maxcur, axis=0)
            scene_maxgrd = np.stack(arrays_maxgrd, axis=0)

            np.save('../../data/3-8/den/picture_%06d.npy'%(imagenum),scene_den)
            np.save('../../data/3-8/maxcur/picture_%06d.npy'%(imagenum),scene_maxcur)
            np.save('../../data/3-8/maxgrd/picture_%06d.npy'%(imagenum),scene_maxgrd)
            # plot 2d to show box and data
            '''
            for kk in layer.keys():
                plot_2d_pc_bbox(scene_den[kk-4], xmin, ymin, xmax, ymax, kk, 'den')
                plot_2d_pc_bbox(scene_maxcur[kk-4], xmin, ymin, xmax, ymax, kk, 'maxcur')
                plot_2d_pc_bbox(scene_maxgrd[kk-4], xmin, ymin, xmax, ymax, kk, 'maxgrd')
            '''
    #return grid

if __name__ == '__main__':

    data = h5py.File('/media/disk2/3D/understanding/rankpooling/rgbd_data/nyu_v2_labeled/nyu_depth_v2_labeled.mat')

    #calibration parameters from author
    K = np.array([[5.7616540758591043e+02,0,3.2442516903961865e+02],[0,5.7375619782082447e+02,2.3584766381177013e+02],[0,0,1]]) 

    count=1;

    start_time = time.time()
    for imagenum in xrange(1,1450):#xrange(data['depths'].shape[0]+1):#size(depths,3):
        #if np.where(np.array([88, 179, 368, 390, 650])==imagenum)[0].size != 0:
        #    continue
        # bed=157, chair=5, table=19, sofa=83, toilet=124
        try:
            box_pc = sio.loadmat('alignData_with_nan/image%04d/annotation_pc.mat' % (imagenum)); # pc generate by ../seeAlignment_pc_3dBox.m
        except:
            continue
        #points3d = points3d'
        pc = box_pc['points3d']; clss=box_pc['clss'][0]
        # change bbox to y,x,z
        ymin = box_pc['ymin'][0]; ymax=box_pc['ymax'][0]; xmin=box_pc['xmin'][0]; xmax=box_pc['xmax'][0]; zmin=box_pc['zmin'][0]; zmax=box_pc['zmax'][0];
        #xmin = box_pc['ymin'][0]; xmax=box_pc['ymax'][0]; ymin=box_pc['xmin'][0]; ymax=box_pc['xmax'][0]; zmin=box_pc['zmin'][0]; zmax=box_pc['zmax'][0];
        pc = np.swapaxes(pc, 0, 1)
        # change pc to y,x,z
        #pc = pc[[1,0,2],:]
        #normals = sio.loadmat('./normalAndpc/normalAndpc%06d.mat'%(imagenum))['normals']

        #if np.where(clss=='chair')[0].size == 0:
        #    continue
        print 'now at image: %d' % (imagenum)
        print 'count: %d' %(count)
        # filter for chair
        #cha_fil = np.where(clss=='chair')
        #xmin = xmin[cha_fil]; ymin = ymin[cha_fil]; zmin = zmin[cha_fil]
        #xmax = xmax[cha_fil]; ymax = ymax[cha_fil]; zmax = zmax[cha_fil]
        #clss = clss[cha_fil]

        # rescale pc
        idxx, idxy, lpc_id = rescale_pc(pc)

        # rescale box to fit image size
        xmin, ymin, xmax, ymax = rescale_box(xmin, ymin, xmax, ymax, pc, idxx, idxy)

        #reduct dimension to 2D    
        to2D(pc, imagenum, xmin, ymin, xmax, ymax, zmin, zmax, idxx, idxy, lpc_id, 'projecting', imagenum)
        '''
        fid = open('../../data/label_5_rcnn/picture_%06d.txt'%(imagenum),'w')
        for k in xrange(len(clss)):
            fid.write('(%d, %d) - (%d, %d) - (%s)\n'%(xmin[k], ymin[k], xmax[k], ymax[k], str(clss[k][0])))
        fid.close()
        fid = open('../../data/label_5_clsfy/picture_%06d.txt'%(imagenum),'w')
        for k in xrange(len(clss)):
            fid.write('%d %d %d %d %s\n'%(xmin[k], ymin[k], xmax[k], ymax[k], str(clss[k][0])))
        fid.close()
        '''
        count+=1
    print 'time: %.2f s' % (time.time()-start_time)
