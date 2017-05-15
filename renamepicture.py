import os
import pdb

typeimg = 'rgb'
formatimg = 'png'
path = '/home/closerbibi/workspace/data/%sbv/projecting_%s_5001'%(typeimg, formatimg);
for filename in os.listdir(path):
    if filename.startswith("picture"):
        wrongnum = filename.split('_')[1].split('.')[0]
        rightnum = int(wrongnum)+5000
        rightname = 'img_%04d.png' % rightnum
        oldname = '/home/closerbibi/workspace/data/%sbv/projecting_%s_5001/%s' %(typeimg, formatimg, filename)
        newname = '/home/closerbibi/workspace/data/%sbv/projecting_%s_5001/%s' %(typeimg, formatimg, rightname)
        os.rename(oldname, newname)
