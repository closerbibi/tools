import numpy as np
import cv2
import os
import pdb

source = '/home/closerbibi/workspace/data/hhabv/projecting'
target = '/home/closerbibi/workspace/data/hhabv/projecting_jpg'

for fname in os.listdir(source):
    grid = np.load(os.path.join(source,fname))
    grid1 = np.swapaxes(grid,0,2)
    img = np.swapaxes(grid1,0,1)
    gen_img = os.path.join(target,fname.split('.')[0])
    new_path_name = '%s.jpg'%(gen_img)
    cv2.imwrite(new_path_name,img)



