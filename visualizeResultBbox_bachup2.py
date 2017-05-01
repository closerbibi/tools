import pdb
import scipy.io as sci
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
import pylab
import matplotlib.cm as cm
import numpy as np
import cv2

resultpath = '../faster-rcnn_hha/results/train/chair_hha_4w.txt' # train
#resultpath = '../results/test/comp4-27463_det_test_toilet.txt'
gt_path = '../faster-rcnn_hha/data/DIRE/Annotations/'
sceneimgpath = '../data/NYUdata/'
#layerimgpath = '/home/closerbibi/bin/faster-rcnn/data/DIRE/ImagesNoRankPooling/'
layerimgpath = '../bin/faster-rcnn/SUNRGBD/SUNRGBDtoolbox/SUNRGBDtoolbox/slicedData/picture_forPooling/'
#layerimgpath = '/home/closerbibi/bin/faster-rcnn/SUNRGBD/SUNRGBDtoolbox/SUNRGBDtoolbox/slicedData/picture_forPooling/'
prenum = 0
#keys=sci.loadmat('/home/closerbibi/bin/faster-rcnn/data/DIRE/imgIDmatch.mat')['keySet']
#len=1143, final num=1449

'''
new index:  Ex: 1,3,4,..,6,9,14... skip some frames in dataset because theres is no target instance in certain frame.
layer index: follow new index and multiplied by 25
result: using new index.
'''
#plt.gca().invert_yaxis()

with open(resultpath) as f:
    for line in f:
        print line
        if float(line.split(' ')[1]) < 0.99:
            continue
        newindex = int(line.split('_')[1].split(' ')[0])
        ttt = line.split(' ')[0].split('_')
        gtfilename = gt_path + ttt[0]+'_'+ttt[1] + '.txt'

        #####################
        if prenum != newindex:

            pdb.set_trace()
            # close previous img
            plt.close('all')

            # find layer image file
            layer = 22#raw_input("Type desired layer here:")
            layerimg =  sci.loadmat(layerimgpath + 'bv_%06d.mat' % (newindex))['grid_whole_img']
            layerimg = np.flipud(layerimg)

            # show current image(layer and scene)
            sceneimgname = sceneimgpath + 'NYU%04d' % (newindex) + '/image/' + 'NYU%04d.jpg' % (newindex)
            #sceneimg = mpimg.imread(sceneimgname)
            sceneimg = cv2.imread(sceneimgname)
            plt.imshow(sceneimg)
            plt.draw()
            plt.show(block=False)

        #continue # just wanna see the gt box and the scene, skip the prediction result
        #continue
        # show current box
        fig2 = pylab.figure()
        ax2 = fig2.add_subplot(111, aspect='equal')
        xmin= float(line.split(' ')[2]); ymin= float(line.split(' ')[3])
        xlen= float(line.split(' ')[4])-float(line.split(' ')[2])
        ylen= float(line.split(' ')[5].split('\n')[0])-float(line.split(' ')[3])
        ax2.add_patch(
            patches.Rectangle(
                (xmin,ymin), xlen, ylen, fill=False, # remove background 
                edgecolor='white'
            )
        )
        # show gt box
        # read txt file
        gtfile = open(gtfilename, 'r')
        target_class = resultpath.split('_')[3].split('.')[0]
        for gt_line in gtfile:
            current_class = gt_line.split('(')[3].split(')')[0];
            if not current_class == target_class:
                continue
            ax3 = fig2.add_subplot(111, aspect='equal')
            xmin= float(gt_line.split('(')[1].split(',')[0]); ymin= float(gt_line.split(')')[0].split(' ')[1])
            xmax= float(gt_line.split('(')[2].split(',')[0]); ymax= float(gt_line.split(')')[1].split(' ')[3])
            xlen=  xmax - xmin
            ylen=  ymax - ymin
            ax3.add_patch(
                patches.Rectangle(
                    (xmin,ymin), xlen, ylen, fill=False, # remove background 
                    edgecolor='green'
                )
            )
        plt.imshow(layerimg)
        plt.title('score = %f' % (float(line.split(' ')[1])))
        plt.draw()
        #plt.gca().invert_yaxis()
        plt.show(block=False)
        # print current score
        #pre_num = newindex
        
        prenum = newindex
    pdb.set_trace()
