import numpy as np
import lmdb
import caffe
import os
import pdb
import cv2
import shutil
import scipy.io as sio
import scipy.misc as smi
import matplotlib.pyplot as plt

def find_mean(X):
    total_sum = np.sum(np.sum(np.sum(X,axis=0),axis=1),axis=1)
    pixel_count = X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]
    print total_sum/pixel_count


def loadtoXY(path, classdict):
    filelist = np.sort(os.listdir(path))
    N = len(filelist)
    X = np.zeros((N, 6, 256, 256)) # from uint8 turn into int64
    Y = np.zeros(N, dtype=np.int64)

    # loading data and label
    count = 0
    nextloop = 0
    for fname in filelist:
        tmpX = np.load(path+'/'+fname)
        tmpSplit={}
        for ch in xrange(6):
            try:
                tmpSplit[ch] = smi.imresize(tmpX[ch],(256,256))
            except:
                nextloop = 1
                break 
        if nextloop == 1:
            print fname
            count += 1
            nextloop = 0
            continue
        array_tmp = [tmpSplit[ser] for ser in tmpSplit.keys()]
        X[count] = np.stack(array_tmp, axis=0)

        # label
        Y[count] = classdict[fname.split('_')[0]]
        #print 'file %s done' % fname
        '''
        for kkk in xrange(6):
            plt.imshow(X[count][kkk])
            plt.show()
        '''
        count += 1
        if count == N:
            break
    return X, Y, N

def creat_lmdb(option, X, Y, N):
    # We need to prepare the database for the size. We'll set it 10 times
    # greater than what we theoretically need. There is little drawback to
    # setting this too big. If you still run into problem after raising
    # this, you might want to try saving fewer entries in a single
    # transaction.
    map_size = X.nbytes * 10

    env = lmdb.open(option+'_lmdb', map_size=map_size)

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N):
            datum = caffe.proto.caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tobytes()  # or .tostring() if numpy < 1.9
            datum.label = int(Y[i])
            str_id = '{:08}'.format(i)
            # The encode is only essential in Python 3
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

if __name__ == '__main__':
    #train_path = '/home/closerbibi/3D/understanding/rankpooling/python/train_dir'
    #test_path = '/home/closerbibi/3D/understanding/rankpooling/python/test_dir'
    train_path = '../../data/clsfy/npydata'


    if os.path.exists('train_lmdb'):
        shutil.rmtree('train_lmdb')
    if os.path.exists('test_lmdb'):
        shutil.rmtree('test_lmdb')
    os.makedirs('train_lmdb')
    os.makedirs('test_lmdb')


    classdict = {
            'chair': 1,
            }
    #X,Y,N = loadtoXY('train', classdict)
    #creat_lmdb('train', X, Y, N)
    X,Y,N = loadtoXY(train_path, classdict)
    creat_lmdb('train', X, Y, N)
    find_mean(X)
