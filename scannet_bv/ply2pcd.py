import sys, os, pdb
sys.path.append('/home/closerbibi/bin/python-pcl')
import pcl

path = '/media/disk3/data/scannet/'


lst = sorted(os.listdir(path))

for f in lst:
    if '_00' in f:
        print('now at {}'.format(f))
        pc = pcl.load(path+f+'/'+f+'_vh_clean_align.ply','ply')
        pcl.save(pc, path+f+'/'+f+'_vh_clean_align.pcd', 'pcd')
