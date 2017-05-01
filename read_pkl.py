import pdb
import pickle

myfile = '../faster-rcnn_3-8/data/cache/train_gt_roidb.pkl'
objects = []
with (open(myfile, "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
pdb.set_trace()
