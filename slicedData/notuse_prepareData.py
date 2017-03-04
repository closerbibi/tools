# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:36:15 2017

@author: closerbibi
"""

import scipy.io as sio
import os
import numpy as np

data = sio.loadmat('/media/closerbibi/internal3/3D/understanding/rankpooling/rgbd_data/nyu_v2_labeled/nyu_depth_v2_labeled.mat')

#calibration parameters from author
K = np.array([[5.7616540758591043e+02,0,3.2442516903961865e+02],[0,5.7375619782082447e+02,2.3584766381177013e+02],[0,0,1]]) 

count=1;

for imagenum in size(depths,3):
    # bed=157, chair=5, table=19, sofa=83, toilet=124
    target=find(labels(:,:,imagenum)==157 | labels(:,:,imagenum)==5 | labels(:,:,imagenum)==19 | labels(:,:,imagenum)==83 | labels(:,:,imagenum)==124);
    if isempty(target)
        continue
    end        
	XYZcamera = depth2XYZcamera(K, depths(:,:,imagenum));
	norfile=sprintf('./feature/normal/norNpoint_%06d.mat',imagenum);
	if ~exist(norfile)
    	[normals,points3D,label3D,imgidx,instance3D] = ...
            depth2normal(XYZcamera,10,labels(:,:,imagenum),[],0,instances(:,:,imagenum));
		save(norfile,...
		'normals','points3D','label3D','imgidx')
	else
		load(norfile)
	end	
	maxdepth = max(max(depths(:,:,imagenum)));
    instance3D = availableInstance(XYZcamera, instances(:,:,imagenum));
    [grid, gridlabel2D] = ...
			slicedto2D(accelData(imagenum,:), points3D, label3D, normals, imgidx, maxdepth, count, instance3D);
        
% %     save(sprintf('./feature/target_class_only/picture/picture_%d.mat',count),...
% % 	 'grid2D', 'gridnormalx', 'gridnormaly', 'gridnormalz', 'label2D');


    keySet = [keySet imagenum];
    valueSet = [valueSet count];
    count=count+1;
	toc;
    imagenum
