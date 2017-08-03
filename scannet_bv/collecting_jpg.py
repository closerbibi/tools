import os, pdb
from shutil import copyfile


src_path = '/media/disk3/data/scannet/'
dst_path = '/home/closerbibi/workspace/data/scannet/rgbbv_align/'

flst = sorted(os.listdir(src_path))

for f in flst:
    if '_00' in f:
        try:
            copyfile(src_path+f+'/grid_rgb_dense_align.jpg', dst_path+f)
            print(f)
        except:
            continue

