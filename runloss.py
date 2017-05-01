import numpy as np
import parseLoss as pl
import pdb
import matplotlib.pyplot as plt
import matplotlib as mt
import collections

#mt.use('Agg')
#plt.ioff()

def loadfile(filename):
    with open(filename, 'r') as infile:
        line = ''
        iter_dict = {}
        while True:
            while 'Iteration' not in line:
                try:
                    line = next(infile)
                    continue
                except:
                    return iter_dict
            entry = {}
            set_index = 5
            iteration = line.split(' ')[set_index].split(',')[0] #5-->6
            string = line.split(' ')[set_index+1] #6-->7
            print iteration
            print string
            try:
                if (int(iteration) % 40 == 0) and (string == 'loss'):
                    loss = line.split(' ')[set_index+3].split('\n')[0] #8 --> 9
                    loss = float(loss)
                    print loss
                    #if loss > 3.0:
                    #    loss = 3.0
                    iter_dict[int(iteration)] = loss
            except ValueError:
                print 'value error'
            line = next(infile)  

if __name__ == '__main__':
    logname = 'train-vgg-hha.log' # remember to change this
    path = '%s' % logname
    iter_dict = loadfile(path)
    iter_dict = collections.OrderedDict(sorted(iter_dict.items()))
    fig = plt.figure()
    #plt.bar(2000*np.array(range(len(iter_dict.keys()[0:-1:50]))), iter_dict.values()[0:-1:50], align='center')
    num = 20
    plt.plot(iter_dict.keys()[0:-1:20], iter_dict.values()[0:-1:20])
    #plt.xticks(range(len(iter_dict.keys()[0:-1:50])), iter_dict.keys()[0:-1:50])
    plt.ylabel('loss')
    plt.xlabel('iteration')
    fig.savefig('loss_%s.png'% logname)
    #fig.savefig('~/Dropbox/schoolprint/lab/meeting/meeting19092016/loss_lrdot05.png')
