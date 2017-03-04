import json
import pdb

for count in xrange(3):
    path = '/home/closerbibi/bin/faster-rcnn/SUNRGBD/kv1/NYUdata/NYU%04d/annotation2Dfinal/index.json' % (count+1)
    with open(path) as json_data:
        d = json.load(json_data)
    pdb.set_trace()
