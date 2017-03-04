import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

pc = np.load('train_dir/sofa_000000585_08.npy')
inds = np.where(pc==1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
c='r';m='o';
xs = inds[0]
ys = inds[1]
zs = inds[2]
pdb.set_trace()
#ax.scatter(xs, ys, zs, c=c, marker=m)
ax.scatter(xs[0:700], ys[0:700], zs[0:700], c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
