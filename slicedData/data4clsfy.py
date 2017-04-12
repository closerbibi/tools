import numpy as np
import os
import matplotlib.pyplot as plt
import pdb


img_path = '../../data/3-8/maxcur/'
lab_path = '../../data/label_for_clsfy/'


for fname in os.listdir(img_path):
    # get image
    img = np.load(img_path + fname)
    #img = np.load(img_path + 'picture_000001.npy')
    
    label_name = lab_path + fname.split('.')[0] + '.txt'
    #label_name = lab_path + 'picture_000001.txt'
    # get label
    with open(label_name,'r') as lname:
        xmin={};ymin={};xmax={};ymax={};cls={}
        count = 0
        for line in lname:
            tmp = line.split(' ')
            xmin[count] = int(tmp[0])
            ymin[count] = int(tmp[1])
            xmax[count] = int(tmp[2])
            ymax[count] = int(tmp[3])
            cls[count] = tmp[4].split('\n')[0]
            count += 1

    # crop image with bbox 
    obj = {}
    for k in xrange(count):
        obj[k] = img[:,ymin[k]:ymax[k]+1,xmin[k]:xmax[k]+1]
        # verify image
        '''
        for kk in xrange(6):
            plt.imshow(obj[k][kk])
            plt.title('img: %s, obj: %d, layer: %d'%(fname, k, kk))
            plt.show()
        '''
        # save new image
        sname ='../../data/clsfy/npydata/%s_%s_%d.npy'%(cls[0], fname.split('.')[0].split('_')[1], k)
        np.save(sname, obj[k])
    print '%s finished'%(fname)

