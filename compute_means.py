from os import listdir
from os.path import isfile, join
import scipy.io as sio
import numpy as np
import pdb

mypath = '/home/closerbibi/bin/faster-rcnn/data/DIRE/Images'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
means0=0.0; means1=0.0;means2=0.0;means3=0.0;

for count in xrange(len(onlyfiles)):
    name = '/home/closerbibi/bin/faster-rcnn/data/DIRE/Images/picture_%06d.mat' % (count+1)
    img = sio.loadmat(name)
    means0 += img['grid'][:,:,0].mean()
    means1 += img['grid'][:,:,1].mean()
    means2 += img['grid'][:,:,2].mean()
    means3 += img['grid'][:,:,3].mean()
    print count

means0 = means0/len(onlyfiles)
means1 = means1/len(onlyfiles)
means2 = means2/len(onlyfiles)
means3 = means3/len(onlyfiles)
print (means0,means1,means2,means3)
pdb.set_trace()
