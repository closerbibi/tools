import numpy as np
import os
from matplotlib import pyplot as plt
import pdb

datapath = '../faster-rcnn_2_channel/data/DIRE/Images'

layer=[]
for fname in os.listdir(datapath):
    tmp = int(fname.split('.')[0].split('_')[-1])
    layer.extend([tmp])


uni = list(set(layer))

# fixed bin size
bins = np.arange(0,26,1)

plt.title('hist of layer, chair')
plt.hist(layer, bins=bins)
plt.xlabel('layer')
plt.ylabel('number')

plt.show()


