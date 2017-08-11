import numpy as np
import os, pdb, operator
import matplotlib.pyplot as plt

cls = np.load('cls_stats.npy').item()
cls.pop('nobox')
vcnt = 50
tmp_cls = sorted(cls.items(), key=operator.itemgetter(1))
sorted_cls = {}
name = []
for i in xrange(len(tmp_cls)):
    sorted_cls[i] = tmp_cls[len(tmp_cls)-1-i][1]
    name.append(tmp_cls[len(tmp_cls)-1-i][0])
#final_cls = { key: value for key, value in sorted_cls.items() if value > vcnt and key != 'nobox'}
final_cls = sorted_cls

#name = final_cls.keys()
x = np.array(range(len(name)))
plt.xticks(x, name)

y = np.asarray(final_cls.values())
pdb.set_trace()
plt.figure(1)
plt.plot(x[:20],y[:20])
plt.figure(2)
plt.xticks(x, name)
plt.plot(x[20:40],y[20:40])
plt.figure(3)
plt.xticks(x, name)
plt.plot(x[40:60],y[40:60])
#plt.plot(x,y)
#plt.hist(x[:10],y[:10])
plt.show()

