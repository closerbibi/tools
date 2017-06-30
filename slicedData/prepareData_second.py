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
from functools import partial
def second_largest(numbers):
    count = 0
    m1 = m2 = float('-inf')
    for x in numbers:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2 if count >= 2 else None

def occlusion(lpc_id, cam_center):
    # create occlusion map
    occu = np.zeros(np.shape(lpc_id))
    # loop over each occupied point
    for i in xrange(len(lpc_id)):
        # create ray emmit from camera
        ray = lpc_id[:,i] - cam_center
        ray /= np.linalg.norm(ray) # normalize, make it vetor
        # assign passby point to occluded (in pc lpc_id range, and not occupied)

    print ""

def rescale_pc(pc):
    pcx_shift = pc[0]-np.nanmin(pc[0])
    pcy_shift = pc[1]-np.nanmin(pc[1])
    pcz_shift = pc[2]-np.nanmin(pc[2])
    largex = np.floor(pcx_shift*100)
    largey = np.floor(pcy_shift*100)
    largez = np.floor(pcz_shift*100)
    lpc_id = np.vstack((largex,largey,pc[2],range(len(pc[2]))))
    # new voxelized z point cloud
    #lpc_id = np.vstack((largex,largey,largez,range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)
    cam_center = np.array([np.nanmin(pc[0]), np.nanmin(pc[1]), np.nanmin(pc[2])])
    return idxx,idxy,lpc_id#,cam_center


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

def constructing_grid_pj_prune_roof(max_idxy,idxx,idxy,grid,lpc_id,img_idx):

    ### color image or hha image, both are (2,1,0)
    hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
    img = cv2.imread(hha_name)
    #rgb_name = '/home/closerbibi/workspace/data/NYUimg_only/NYU%04d.jpg'%(int(img_idx))
    #img = cv2.imread(rgb_name)
    #gray_image = cv2.cvtColor(imgrgb, cv2.COLOR_BGR2GRAY)
    disparity = img[:, :, 2] # gray
    height = img[:,:,1] # hha: height
    angle = img[:,:,0] # hha: angle

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
            # channel 0: disparity
            # channel 1: height
            # channel 2: angle
            if len(xy_eligible) > 0:
                # then find the position of max z to complete the map, bv can only see the highest point
                z_lst = lpc_x_sort[2,x_loc][0,xy_eligible]
                z_lst_argsort = (np.argsort(z_lst))[::-1]# sort=small>large, use[::-1] to large>small
                for i in range(len(z_lst_argsort)):
                    original_idx = xy_eligible[z_lst_argsort[i]]
                    #original_idx1 = xy_eligible[np.argmax(z_lst)]
                    location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                    #location1 = lpc_x_sort[3,x_loc][0, original_idx1].astype(int)
                    wmap = int(np.floor(location/hh))
                    hmap = (location%hh)-1
                    if angle[hmap,wmap] < 190: # pruning roof base on angle
                        grid[0][rviy][ix] = angle[hmap,wmap]
                        grid[1][rviy][ix] = height[hmap,wmap]
                        grid[2][rviy][ix] = disparity[hmap,wmap]
                        break
                    else:
                        continue
                    #    grid[0][rviy][ix] = 0
                    #    grid[1][rviy][ix] = 0
                    #    grid[2][rviy][ix] = 0
            else:
                grid[0][rviy][ix] = 0
                grid[1][rviy][ix] = 0 #np.nanmin(height)
                grid[2][rviy][ix] = 0
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
                grid[0][rviy][ix] = disparity[hmap,wmap]
                grid[1][rviy][ix] = height[hmap,wmap]
                grid[2][rviy][ix] = angle[hmap,wmap]
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
    xs = pc[0][range(0,len(pc[0]),10)]
    ys = pc[1][range(0,len(pc[1]),10)]
    zs = pc[2][range(0,len(pc[2]),10)]
    ax.scatter(xs, ys, zs, c=color, marker='o')
    ax.scatter(xmin, ymin, zmin, c='y', marker='^')
    ax.scatter(xmax, ymax, zmax, c='k', marker='^')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def to2D(pc, imagenum, idxx, idxy, lpc_id, method, img_idx, zmin, zmax):
    points_height = pc[2,:]
    ceiling = np.nanmax(points_height)
    floor = np.nanmin(points_height)

    # seperate to multiple layers
    layers = 50 # 25
    l_step = (ceiling - floor)/layers
    layer = {}

    data_grid = np.meshgrid(idxx,idxy)
    s_den = {}
    s_maxcur = {}
    s_maxgrd = {}
    s_normal = {}
    sw=0

    if method == 'projecting':
        # rgb
        #grid_file_2ch = '../../data/hhabv/2ch/picture_%06d.npy'%(imagenum)
        grid_file_hhagray = '../../data/hhabv/remove_roof/picture_%06d.jpg'%(imagenum)
        if not os.path.exists('../../data/hhabv/remove_roof'):
            os.makedirs('../../data/hhabv/remove_roof')
        # hha
        #grid_file = '../../data/hhabv/projecting_noroof_all/picture_%06d.npy'%(imagenum)
        grid = np.zeros((3,len(idxy),len(idxx)))
        max_idxy = np.nanmax(idxy)
        grid = constructing_grid_pj_prune_roof(max_idxy,idxx,idxy,grid,lpc_id,img_idx)
        #np.save(grid_file_2ch,grid[1:])
        #np.save(grid_file_hhagray,grid)
        grid = np.transpose(grid, (1,2,0))
        #plt.imshow(grid), plt.show()
        #pdb.set_trace()
        cv2.imwrite(grid_file_hhagray, grid)
        #else:
        #    print "grid file done."
        
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

                #np.save(grid_file,scene_den)
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
        grid = np.zeros((13,len(idxy),len(idxx)))
        max_idxy = np.nanmax(idxy)

        ### color image or hha image, both are (2,1,0)
        hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
        img = cv2.imread(hha_name)
        angle = img[:,:,0] # hha: angle

        ### find index in the range
        ### according to the analyse, picking layers who are most polular
        start = 0
        for ilayer in xrange(start, start+13):#xrange(layers):
            cur_ceiling = floor + l_step*(ilayer+1)
            cur_floor = floor + l_step*ilayer
            ### trim layer without target class (chair only first)
            ### change to more accurate layer finding
            #if (cur_ceiling > zmin).all() and (cur_floor < zmax).all():
            #    dict_box[ilayer] = dict_box[ilayer] + 1
            idx = np.where((pc[2,:] > cur_floor) * (pc[2,:] < cur_ceiling))
            layer[ilayer] = lpc_id[:,idx[0]]

            ### show 3d data and bbox corner
            #plot_3d(layer, xmin, ymin, xmax, ymax, zmin, zmax, layers) 
            ### making grid
            large_layerpc = layer[ilayer]
            grid_ly = np.zeros((3,len(idxy),len(idxx)))
            max_idxy = np.nanmax(idxy)

            grid_ly = constructing_grid_ly(max_idxy,idxx,idxy,grid_ly,large_layerpc,cur_floor,pc,angle)
            grid[ilayer-start] = grid_ly[0]

        ### debug
        #for aa in xrange(0,12):
        #    plt.imshow(grid[aa])
        #    plt.title(aa)
        #    plt.colorbar()
        #    plt.show()
        #np.save('../../data/ch13_classes19/picture_%06d.npy'%(imagenum),grid)
            # plot 2d to show box and data
        '''
        for kk in layer.keys():
            plot_2d_pc_bbox(scene_den[kk-4], xmin, ymin, xmax, ymax, kk, 'den')
        '''
    if method == 'hybrid':
        # construct disparity and height channel first
        grid = np.zeros((8,len(idxy),len(idxx)))
        grid_3ch = np.zeros((3,len(idxy),len(idxx)))
        max_idxy = np.nanmax(idxy)
        grid_file = '../../data/hhabv/projecting_noroof/picture_%06d.npy'%(imagenum)
        if not os.path.exists(grid_file):
            grid_3ch = constructing_grid_pj_prune_roof(max_idxy,idxx,idxy,grid,lpc_id,img_idx)
        else:
            grid_3ch = np.load(grid_file)
        grid[6] = grid_3ch[0]
        grid[7] = grid_3ch[2]

        ### color image or hha image, both are (2,1,0)
        hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
        img = cv2.imread(hha_name)
        #angle = img[:,:,0] # hha: angle
        height = img[:,:,1] # hha: height

        ### find index in the range
        ### according to the analyse, picking layers who are most polular
        start = 2
        for ilayer in xrange(start, start+6):#xrange(layers):
            cur_ceiling = floor + l_step*(ilayer+1)
            cur_floor = floor + l_step*ilayer
            ### trim layer without target class (chair only first)
            ### change to more accurate layer finding
            #if (cur_ceiling > zmin).all() and (cur_floor < zmax).all():
            #    dict_box[ilayer] = dict_box[ilayer] + 1
            idx = np.where((pc[2,:] > cur_floor) * (pc[2,:] < cur_ceiling))
            layer[ilayer] = lpc_id[:,idx[0]]

            ### show 3d data and bbox corner
            #plot_3d(layer, xmin, ymin, xmax, ymax, zmin, zmax, layers) 
            ### making grid
            large_layerpc = layer[ilayer]
            grid_ly = np.zeros((3,len(idxy),len(idxx)))
            max_idxy = np.nanmax(idxy)

            grid_ly = constructing_grid_ly(max_idxy,idxx,idxy,grid_ly,large_layerpc,cur_floor,pc, height)
            grid[ilayer-start] = grid_ly[0]

        ### debug
        #for aa in xrange(0,8):
        #    plt.imshow(grid[aa])
        #    plt.title(aa)
        #    plt.colorbar()
        #    plt.show()
        #np.save('../../data/layer6_disparity_angle/picture_%06d.npy'%(imagenum),grid)

def runrun(imagenum):
    #if np.where(np.array([88, 179, 368, 390, 650])==imagenum)[0].size != 0:
    #    continue
    # bed=157, chair=5, table=19, sofa=83, toilet=124
    imagenum = int(imagenum)
    box_pc = sio.loadmat('alignData_with_nan_19_classes/image%04d/annotation_pc.mat' % (imagenum)); # pc generate by bin.....SUN3Dtoolbox/generate_pc_3dBox.m
    pc = box_pc['points3d']; 
    pc = np.swapaxes(pc, 0, 1)
    print 'now at image: %d' % (imagenum)

    # rescale pc
    idxx, idxy, lpc_id = rescale_pc(pc)

    clss=box_pc['clss'][0]
    ymin = box_pc['ymin'][0]; ymax=box_pc['ymax'][0]; xmin=box_pc['xmin'][0]; xmax=box_pc['xmax'][0]; zmin=box_pc['zmin'][0]; zmax=box_pc['zmax'][0];
    # rescale box to fit image size
    xmin, ymin, xmax, ymax, zmin, zmax, clss = rescale_box(xmin, ymin, xmax, ymax, zmin, zmax, pc, idxx, idxy, clss, imagenum)
    clss[0]
        
    #reduct dimension to 2D    
    to2D(pc, imagenum, idxx, idxy, lpc_id, 'projecting', imagenum, zmin, zmax)
    #fid = open('../../data/fg/picture_%06d.txt'%(imagenum),'w')
    #fid = open('../../data/label_pascal/%06d.txt'%(imagenum),'w') # for pascal voc format
    #for k in xrange(len(clss)):
    #    try:
    #        if (clss[k][0]) == 'nightstand':
    #            fid.write('(%d, %d) - (%d, %d) - (night_stand)\n'%(xmin[k], ymin[k], xmax[k], ymax[k]))
    #        else:
    #            fid.write('(%d, %d) - (%d, %d) - (%s)\n'%(xmin[k], ymin[k], xmax[k], ymax[k], clss[k][0]))
    #    except:
    #        print 'Ooooooops'
    #fid.close()
    '''
    fid = open('../../data/label_gupta/picture_%06d.txt'%(imagenum),'w')
    for k in xrange(len(clss)):
        fid.write('%d %d %d %d %s\n'%(xmin[k], ymin[k], xmax[k], ymax[k], str(clss[k][0])))
    fid.close()
    '''

if __name__ == '__main__':

    newidx = np.array([89,122,146,149,153,159,254,449,725,755,790,936,1036,1038,1273,1427])
    #dict_box = {k: 0 for k in xrange(25)}
    #fid_nobox = open('/home/closerbibi/workspace/pytorch/faster-rcnn_19classes/data/DIRE/nobox_image.txt','w')
    fid_nobox = open('all_cls.txt','w')
    pool = Pool( processes=4 )
    lst = range(1,1450)
    lst = map(str, lst)
    ah = open('nobox_image.txt', 'r');bah=ah.read();aah=bah.split('\n')[:-1]
    lst = [i for j, i in enumerate(lst) if i not in aah]
    #runrun('1')
    pool.map(runrun, lst)
    pool.close()
    pool.join()
    fid_nobox.close()