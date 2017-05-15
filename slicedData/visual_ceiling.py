import cv2, numpy as np, matplotlib.pyplot as plt
import pdb
for img_idx in xrange(1,1450):
    rgb_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
    img = cv2.imread(rgb_name)
    disparity = img[:,:,2] # hha: disparity                                                                                                                                                                  
    height = img[:,:,1] # hha: height
    angle = img[:,:,0] # hha: angle
    plt.imshow(angle)
    plt.title(img_idx)
    plt.colorbar()
    plt.show()

