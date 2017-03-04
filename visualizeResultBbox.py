import pdb
import scipy.io as sci
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import Image
import pylab
import matplotlib.cm as cm
import numpy as np

resultpath = '../results/train/comp4-12230_det_train_chair.txt'
gt_path = '../data/DIRE/Annotations/'
sceneimgpath = '/home/closerbibi/bin/faster-rcnn/SUNRGBD/kv1/NYUdata/'
#layerimgpath = '/home/closerbibi/bin/faster-rcnn/data/DIRE/ImagesNoRankPooling/'
layerimgpath = '/home/closerbibi/bin/faster-rcnn/SUNRGBD/SUNRGBDtoolbox/SUNRGBDtoolbox/slicedData/picture_forPooling/'
#layerimgpath = '/home/closerbibi/bin/faster-rcnn/SUNRGBD/SUNRGBDtoolbox/SUNRGBDtoolbox/slicedData/picture_forPooling/'
pre_num = 0
#keys=sci.loadmat('/home/closerbibi/bin/faster-rcnn/data/DIRE/imgIDmatch.mat')['keySet']
#len=1143, final num=1449

'''
new index:  Ex: 1,3,4,..,6,9,14... skip some frames in dataset because theres is no target instance in certain frame.
layer index: follow new index and multiplied by 25
result: using new index.
'''
def show_box(line, fig):
    # show current box
    #fig2.gca().invert_yaxis()
    ax = fig.add_subplot(111, aspect='equal')
    xmin= float(line.split(' ')[2]); ymin= float(line.split(' ')[3])
    xlen= float(line.split(' ')[4])-float(line.split(' ')[2])
    ylen= float(line.split(' ')[5].split('\n')[0])-float(line.split(' ')[3])
    ax.add_patch(
        patches.Rectangle(
            (xmin,ymin), xlen, ylen, fill=False, # remove background 
            edgecolor='white'
        )
    )
    return ax

def show_gtbox(gt_line, fig):
    ax = fig.add_subplot(111, aspect='equal')
    xmin= float(gt_line.split('(')[1].split(',')[0]); ymin= float(gt_line.split(')')[0].split(' ')[1])
    xmax= float(gt_line.split('(')[2].split(',')[0]); ymax= float(gt_line.split(')')[1].split(' ')[3])
    xlen=  xmax - xmin
    ylen=  ymax - ymin
    ax.add_patch(
        patches.Rectangle(
            (xmin,ymin), xlen, ylen, fill=False, # remove background 
            edgecolor='green'
        )
    )
    return ax

with open(resultpath) as f:
    for line in f:
        print line
        if float(line.split(' ')[1]) < 0.9:
            continue
        newindex = int(line.split('_')[1].split(' ')[0])
        ###trueindex = keys[0][newindex-1]
        gtfilename = gt_path + line.split(' ')[0] + '.txt'
        if newindex == pre_num:
            #continue # just wanna see the gt box and the scene, skip the prediction result
            # show current box
            fig = pylab.figure()
            ax2 = show_box(line, fig)
            # show gt box
            # read txt file
            gtfile = open(gtfilename, 'r')
            target_class = resultpath.split('_')[3].split('.')[0]
            for gt_line in gtfile:
                current_class = gt_line.split('(')[3].split(')')[0];
                if not current_class == target_class:
                    continue
                ax3 = show_box(gt_line, fig)
            plt.imshow(layerimg)
            plt.title('score = %f' % (float(line.split(' ')[1])))
            plt.draw()
            plt.gca().invert_yaxis()
            plt.show(block=False)
            # print current score
            pre_num = newindex
        else: 
            pdb.set_trace()
            # close previous img
            plt.close('all')

            # show current image(layer and scene)
            sceneimgname = sceneimgpath + 'NYU%04d' % (newindex) + '/image/' + 'NYU%04d.jpg' % (newindex)
            sceneimg = mpimg.imread(sceneimgname)
#            fig1 = pylab.figure()
            plt.imshow(sceneimg)
            plt.draw()
            plt.show(block=False)
            layer = 22#raw_input("Type desired layer here:")
            layerimg =  sci.loadmat(layerimgpath + 'picture_%06d_01.mat' % (newindex))['grid_whole_img']
            # show current box
            fig2 = pylab.figure()
            ax2 = show_box(line, fig2)
            plt.imshow(layerimg)

            # show gt box
            # read txt file
            gtfile = open(gtfilename, 'r')
            target_class = resultpath.split('_')[3].split('.')[0]
            for gt_line in gtfile:
                current_class = gt_line.split('(')[3].split(')')[0];
                if not current_class == target_class:
                    continue
                ax3 = show_gtbox(gt_line, fig2)
                plt.imshow(layerimg)
            plt.title('score = %f' % (float(line.split(' ')[1])))
            plt.draw()
            plt.gca().invert_yaxis()
            plt.show(block=False)
            pre_num = newindex

