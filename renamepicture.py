import os
import pdb

path = '/home/closerbibi/bin/faster-rcnn/data/DIRE/Annotations';
for filename in os.listdir(path):
    if filename.startswith("picture_"):
        wrongnum = filename.spilt('_')[1].split('.')[0]
        rightnum = ((int(wrongnum)-1)/25)+1
        rightname = 'picture_%06d.mat' % rightnum
        os.rename(filename, rightname)
        pdb.set_trace()
