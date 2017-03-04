import parseLoss as pl
import pdb
import matplotlib.pyplot as plt
import matplotlib as mt

#mt.use('Agg')
#plt.ioff()
logname = 'train_rankpooling' # remember to change this
path = '/home/closerbibi/bin/faster-rcnn/logfile/%s.log' % logname
loss = pl.loadfile(path)
fig = plt.figure()
plt.bar(range(len(loss)), loss.values(), align='center')
plt.xticks(range(len(loss)), loss.keys())
plt.ylabel('loss')
plt.xlabel('iteration')
fig.savefig('loss_%s.png'% logname)
#fig.savefig('~/Dropbox/schoolprint/lab/meeting/meeting19092016/loss_lrdot05.png')
