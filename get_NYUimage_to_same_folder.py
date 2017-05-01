import os
import numpy as np
import shutil
import pdb



path = '/home/closerbibi/workspace/data/NYUdata/'
target = '/home/closerbibi/workspace/data/NYUdepth_bfx_only/'

'''
for fname in os.listdir(path):
    if not 'NYU' in fname:
        continue
    kkk = fname + '.jpg'
    imgpath = path + fname + '/image/' + kkk
    shutil.copyfile(imgpath, target+kkk)
'''

for fname in os.listdir(path):
    if not 'NYU' in fname:
        continue
    kkk = fname + '.png'
    imgpath = path + fname + '/depth_bfx/' + kkk
    shutil.copyfile(imgpath, target+kkk)
