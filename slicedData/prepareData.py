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
from PIL import Image
from numpy.linalg import norm
import numba as nb # vectorize & jit
from occlusion import *
from plot_tool import *
from mapping_grid import *
from rescale_data import *

def hhatobv(lpc_id, hhavalue, imagenum, idxx, idxy):
    # pc: (x, z, -y), z is depth
    grid_file_path = '../../data/hhabv/upsample'
    if not os.path.exists(grid_file_path):
        os.makedirs(grid_file_path)
    gfile = grid_file_path + '/picture_{:06d}.jpg'.format(imagenum)
    grid = np.zeros((3,len(idxy),len(idxx)))
    max_idxy = np.nanmax(idxy)
    if not os.path.exists(gfile):
        grid = hhagrid(lpc_id, hhavalue, imagenum, grid, idxx, idxy)
        img = np.transpose(grid, (1,2,0))
        cv2.imwrite(gfile, img)

def to2D(pc, imagenum, idxx, idxy, lpc_id, method, img_idx, zmin, zmax,occu_map):
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
        #grid_file = '../../data/hhabv/occlu_jpg_v2/picture_%06d.jpg'%(imagenum)
        #grid_file_pascal = '../../data/rgbbv/projecting_noroof_all_pascal_jpg/%06d.jpg'%(imagenum)
        # hha
        grid_file_path = '../../data/hhabv/upsample'
        if not os.path.exists(grid_file_path):
            os.makedirs(grid_file_path)
        gfile = grid_file_path + '/picture_{:06d}.jpg'.format(imagenum)
        grid = np.zeros((3,len(idxy),len(idxx)))
        max_idxy = np.nanmax(idxy)
        if not os.path.exists(gfile):
            grid = constructing_grid_pj_prune_roof(max_idxy,idxx,idxy,grid,lpc_id,img_idx,occu_map)
            img = np.transpose(grid, (1,2,0))
            cv2.imwrite(gfile, img)
            #cv2.imwrite(grid_file_pascal, img)
            #np.save(grid_file_2ch,grid[1:])
            #np.save(grid_file_hhagray,grid)
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

def runhhacloud(imagenum):
    imagenum = int(imagenum)
    print 'now at image: %d' % (imagenum)
    box_pc = sio.loadmat('alignData_with_nan_19_classes/image%04d/annotation_pc.mat' % (imagenum)); # pc generate by bin.....SUN3Dtoolbox/generate_pc_3dBox.m
    # load pc from up sampling
    pc = np.load('../../data/hhaupsample/image{:04d}_pc.npy'.format(imagenum))
    pc = np.swapaxes(pc, 0, 1)
    lpc_id, idxx, idxy = rescale_upsampling_pc(pc)
    hhavalue = np.load('../../data/hhaupsample/image{:04d}_hha.npy'.format(imagenum))

    #occu_map = z_buffer(voxel_world, idxx, idxy, idxz)

    clss=box_pc['clss'][0]
    ymin = box_pc['ymin'][0]; ymax=box_pc['ymax'][0]; xmin=box_pc['xmin'][0]; xmax=box_pc['xmax'][0]; zmin=box_pc['zmin'][0]; zmax=box_pc['zmax'][0];
    # rescale box to fit image size
    xmin, ymin, xmax, ymax, zmin, zmax, clss = rescale_box(xmin, ymin, xmax, ymax, zmin, zmax,
                                                           pc/100., idxx, idxy, clss, imagenum)
    #plot_3d(lpc_id, xmin, xmax, ymin, ymax, zmin, zmax)
    #reduct dimension to 2D    
    hhatobv(lpc_id, hhavalue, imagenum, idxx, idxy)
    #fid = open('../../data/fg/picture_%06d.txt'%(imagenum),'w')
    label_dir = '../../data/label_19_upsample'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    fid = open(label_dir+'/picture_%06d.txt'%(imagenum),'w')
    for k in xrange(len(clss)):
        try:
            if (clss[k][0]) == 'nightstand':
                fid.write('(%d, %d) - (%d, %d) - (night_stand)\n'%(xmin[k], ymin[k], xmax[k], ymax[k]))
            else:
                fid.write('(%d, %d) - (%d, %d) - (%s)\n'%(xmin[k], ymin[k], xmax[k], ymax[k], clss[k][0]))
        except:
            print 'Ooooooops'
    fid.close()



def runrun(imagenum):
    #if np.where(np.array([88, 179, 368, 390, 650])==imagenum)[0].size != 0:
    #    continue
    # bed=157, chair=5, table=19, sofa=83, toilet=124
    imagenum = int(imagenum)
    print 'now at image: %d' % (imagenum)
    box_pc = sio.loadmat('alignData_with_nan_19_classes/image%04d/annotation_pc.mat' % (imagenum)); # pc generate by bin.....SUN3Dtoolbox/generate_pc_3dBox.m
    # load pc from princeton
    pc = box_pc['points3d']; 
    pc = np.swapaxes(pc, 0, 1)


    # rescale pc
    idxx, idxy, idxz, lpc_id, cam_center, voxel_world = rescale_pc_200(pc)

    occu_map = z_buffer(voxel_world, idxx, idxy, idxz)

    clss=box_pc['clss'][0]
    ymin = box_pc['ymin'][0]; ymax=box_pc['ymax'][0]; xmin=box_pc['xmin'][0]; xmax=box_pc['xmax'][0]; zmin=box_pc['zmin'][0]; zmax=box_pc['zmax'][0];
    # rescale box to fit image size
    xmin, ymin, xmax, ymax, zmin, zmax, clss = rescale_box_200(xmin, ymin, xmax, ymax, zmin, zmax, pc, idxx, idxy, clss, imagenum)

    plot_3d(lpc_id, xmin, xmax, ymin, ymax, zmin, zmax)
    pdb.set_trace()
    #reduct dimension to 2D    
    to2D(pc, imagenum, idxx, idxy, lpc_id, 'projecting', imagenum, zmin, zmax,occu_map)
    #fid = open('../../data/fg/picture_%06d.txt'%(imagenum),'w')
    #label_dir = '../../data/label_19_200'
    #if not os.path.exists(label_dir):
    #    os.makedirs(label_dir)
    #fid = open(label_dir+'/picture_%06d.txt'%(imagenum),'w')
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
    lst = range(1,1450)
    lst = map(str, lst)
    ah = open('nobox_image.txt', 'r');bah=ah.read();aah=bah.split('\n')[:-1]
    lst = [i for j, i in enumerate(lst) if i not in aah]
    runhhacloud('1')
    #pool = Pool( processes=4 )
    #pool.map(runhhaclud, lst)
    #pool.close()
    #pool.join()
    fid_nobox.close()


# turn voxel location to array
'''
location = np.zeros((len(idxx)*len(idxy)*len(idxz),3))
count = 0
t0 = time.time()
for i in idxx:
    for jj in idxy:
        for k in idxz:
            location[count] = [i,jj,k]
            count += 1
print time.time()-t0
for num in xrange(voxel_world.shape[1]):
    tmp = voxel_world[:3,0]; tmp = tmp[np.newaxis,:]
    current_pc = np.repeat(tmp, len(idxx)*len(idxy)*len(idxz), axis=0)
    cam_center = cam_center[np.newaxis,:]
    cam_center = np.repeat(cam_center, len(idxx)*len(idxy)*len(idxz), axis=0)
    # distance from current one point cloud point to line between camera center and each voxel
    t1 = time.time()
    # count distance
    p1p2 = cam_center - location;
    normp1p2 = norm(p1p2)
    # line: p1, p2; point: p3i
    # d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)
    dis = norm(np.cross(p1p2, current_pc - location))/normp1p2
    #dis = voxel_dis(current_pc, cam_center, location)
    print time.time()-t1
    pdb.set_trace()
    # find the occlusion voxel depend on distance
'''
