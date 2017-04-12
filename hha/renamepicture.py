import os
import pdb

path = '/home/closerbibi/workspace/tools/hha/results';
for filename in os.listdir(path):
    if filename.startswith("NYU"):
        wrongnum = filename.split('NYU')[1].split('.')[0]
        rightnum = int(wrongnum)+5000
        rightname = 'img_%04d.png' % rightnum
        oldname = '/home/closerbibi/workspace/tools/hha/results/%s' %(filename)
        newname = '/home/closerbibi/workspace/tools/hha/results/%s' %(rightname)
        os.rename(oldname, newname)
