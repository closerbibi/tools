import numpy as np
import cv2, os, sys, pdb, time
from plyfile import PlyData as plyd
from plyfile import PlyElement as plye
from multiprocessing import Pool
import matplotlib.pyplot as plt
sys.path.append('/home/kevin/3D_project/code')
import basic as ba
import scipy.io as sio

def constructing_grid_pj(idxx, idxy, lpc_id, pc, hha2ch):
    grid_rgb = np.zeros((3, len(idxy),len(idxx)))
    #grid_depth = np.zeros((len(idxy),len(idxx)))
    max_idxy = np.nanmax(idxy)
    max_height = np.nanmax(pc[2])
    lpc_id = lpc_id[:,~np.isnan(lpc_id[0])]
    lpc_trans = np.transpose(lpc_id,(1,0))
    lpc_x_sort = lpc_trans[lpc_trans[:, 0].argsort()]
    lpc_x_sort = np.transpose(lpc_x_sort,(1,0))
    for ix in idxx:
        x_loc = np.where(lpc_x_sort[0].astype(int)==ix)
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            xy_eligible = np.where(lpc_x_sort[1][x_loc].astype(int)==iy)[0]
            #xy_eligible = (largexy[0].astype(int)==ix)*(largexy[1].astype(int)==iy) 
            #xy_eligible = np.where(xy_eligible==True)[0]
            if len(xy_eligible) > 0:
                # then find the max z position to complete the map, bv can only see the highest point
                original_idx = xy_eligible[np.argmax(lpc_x_sort[2,x_loc][0,xy_eligible])]
                location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                #location = xy_eligible[np.argmax(pc[2][xy_eligible])]
                #grid_depth[rviy][ix] = max_height - pc[2][location]#depth
                grid_rgb[0][rviy][ix] = hha2ch[location][0]#R
                grid_rgb[1][rviy][ix] = hha2ch[location][1]#G
                #grid_rgb[2][rviy][ix] = rgb[location][2]#B
            else:
                #grid_depth[rviy][ix] = 0
                grid_rgb[0][rviy][ix] = 0
                grid_rgb[1][rviy][ix] = 0 #np.nanmin(height)
                #grid_rgb[2][rviy][ix] = 0
    #plt.imshow(grid_rgb[0]);plt.colorbar();plt.title('height');plt.show()
    #plt.imshow(grid_rgb[1]);plt.colorbar();plt.title('angle');plt.show()
    return grid_rgb#, grid_depth

def axis_2d(axis):
    #return (np.floor(np.array(axis)*100)).tolist()
    return (np.floor(np.array(axis))).tolist()

def gen_box(f_seg, f_par, cls_lst, num):
    segInd = np.array(f_seg['segIndices'])
    label_group = f_par['segGroups']
    object_lst = {}
    for i in range(len(label_group)):
        if not label_group[i]['label'] in object_lst.keys():
            object_lst[label_group[i]['label']] = []
        
        try:
            if not str(label_group[i]['label']) in cls_lst:
                cls_lst[str(label_group[i]['label'])] = 1
            else:
                cls_lst[str(label_group[i]['label'])] += 1
        except:
            cls_lst['nobox'].append(num)
    return cls_lst

def rotateManhattan(pc, num):
    R = sio.loadmat('rotation_m/rotation{:04d}.mat'.format(num))['R']
    pc = np.dot(np.transpose(pc,(1,0)), R)
    pc = np.transpose(pc, (1,0))
    return pc

def vis(gt_boxes, num):
    img_path = '/home/closerbibi/workspace/data/scannet/rgbbv_align/'
    #img_path = '/home/closerbibi/workspace/data/scannet/rgbbv/'
    im = cv2.imread(img_path+'scene{:04d}_00.jpg'.format(num))
    im = im[:,:,(2,1,0)]
    # plot image
    fig = plt.figure(figsize=(20,10))
    img = plt.imshow(im, interpolation='nearest')
    img.set_cmap('hot')
    #plt.axis('off')
    #plt.axis([-500, 1500, -500, 1500])

    # plot box
    for ii in xrange(len(gt_boxes)):
        pdb.set_trace()
        for jj in xrange(len(gt_boxes.values()[ii])):
            x1=gt_boxes.values()[ii][0][0][0]
            y1=gt_boxes.values()[ii][0][1][0]
            x2=gt_boxes.values()[ii][0][0][1]
            y2=gt_boxes.values()[ii][0][1][1]
        try:
            class_name = str(gt_boxes.keys()[ii])
        except:
            return
        plt.gca().add_patch(
            plt.Rectangle((x1, y1),
                            x2 - x1,
                            y2 - y1, fill=False,
                            edgecolor='g', linewidth=0.7)
            )
        plt.gca().text(x1, y1 - 2,
            '{:s}'.format(class_name),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=12, color='white')
    plt.title('%d'%(num))
    #fig.savefig('bv_with_box/{:04d}.jpg'.format(num), dpi=fig.dpi)


def rescale_pc(pc):
    pcx_shift = pc[0]-np.nanmin(pc[0])
    pcy_shift = pc[1]-np.nanmin(pc[1])
    largex = np.floor(pcx_shift*100)
    largey = np.floor(pcy_shift*100)
    #largexy = np.vstack((largex,largey))
    lpc_id = np.vstack((largex,largey,pc[2],range(len(pc[2]))))
    idxx = range(int(np.nanmin(lpc_id[0])),int(np.nanmax(lpc_id[0])+1),1)
    idxy = range(int(np.nanmin(lpc_id[1])),int(np.nanmax(lpc_id[1])+1),1)
    return idxx, idxy, lpc_id


def new_gen_pc_rgb(data):
    pc, rgb = np.zeros((3,len(data))), np.zeros((len(data),3))
    for i in range(len(data)):
        pc[0][i], pc[1][i], pc[2][i] = data[i][0], data[i][1], data[i][2]
        rgb[i][0], rgb[i][1], rgb[i][2] = data[i][3], data[i][4], data[i][5]
    return pc, rgb


def gen_pc_label(data):
    x, y, z, seglabel = [], [], [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
        z.append(data[i][2])
        seglabel.append([data[i][7]])
    x, y, z = np.array(x), np.array(y), np.array(z)
    seglabel = np.array(seglabel)
    pc = np.vstack((x, y, z))
    return pc, seglabel

def parse_label(num):
    data_path = '/media/disk3/data/scannet/scene{:04d}_00/scene{:04d}_00'.format(num, num)
    f_seg = ba.LoadJson(data_path + '_vh_clean_2.0.010000.segs.json')
    f_par = ba.LoadJson(data_path + '.aggregation.json')
    f = plyd.read('{}_vh_clean_2.labels.ply'.format(data_path))
    data = f.elements[0].data
    labelpc, seglabel = gen_pc_label(data)
    return f_seg, f_par, labelpc

def gen_data_bv(ID, cls_lst):
    img_path = '/home/closerbibi/workspace/data/scannet/pcwithnormal/'# for ply
    target_path = '/home/closerbibi/workspace/data/scannet/hhabv2ch/'
    data_path = img_path + '/' + ID
    num = int(ID.split('_')[0].split('image')[1])

    if os.path.isfile(target_path + 'picture_{:06d}.npy'.format(num)):
        #return
        pass
    print 'Start at %s !'%ID
    start = time.time()
    # load for npy
    #data = np.load(img_path+ID)

    # data saperate to pc, hha2ch
    #pc, hha2ch = data[:,0:3], data[:, 3:]
    #pc = pc/100. # meter to centermeter
    #pc = np.transpose(pc,(1,0))
    #idxx,idxy,largexy = rescale_pc(pc)

    # to grid
    t2 = time.time()
    #grid = constructing_grid_pj(idxx, idxy, largexy, pc, hha2ch)
    #print('to grid: {:.2f} secs'.format(time.time()-t2))
    f_seg, f_par, labelpc = parse_label(num)
    # pc input in meter, bbox output in centermeter
    cls_lst = gen_box(f_seg, f_par, cls_lst, num)
    # visualize box on image
    #vis(bbox, num)

    # write to data file
    t3 = time.time()
    #ba.WriteJson(data_path, 'bbox_dense', bbox)
    #cv2.imwrite('verify/h/' + 'picture_{:06d}.jpg'.format(num), grid[0])
    #cv2.imwrite('verify/a/' + 'picture_{:06d}.jpg'.format(num), grid[1])
    #np.save(target_path + 'picture_{:06d}.npy'.format(num), grid)
    
    
    #grid_rgb_img = np.transpose(grid_rgb[(2,1,0),:,:],(1,2,0))
    #cv2.imwrite(data_path + '/grid_depth_dense_align.jpg', grid_depth)
    #print('write: {:.2f} secs'.format(time.time()-t3))
    
    total = (time.time() - start)
    #print '%s finish, time: %.2f secs\n'%(ID, total) 
    return cls_lst



def parse_lst(lst):
    new_lst = []
    for ID in lst:
        new_ID = ID.split('_')[0]
        if not new_ID in new_lst:
            new_lst.append(new_ID)
    return new_lst

def main():
    img_path = '/home/closerbibi/workspace/data/scannet/pcwithnormal'
    img_lst = sorted(os.listdir(img_path))
    cls_lst = {}
    cls_lst['nobox'] = []
    #cls_lst['zero_class'] = []
    #gen_data_bv('image0001_pchha2ch.npy')
    for ID in img_lst:
        cls_lst = gen_data_bv(ID, cls_lst)
    np.save('cls_stats.npy', cls_lst) # load: np.load('cls_stats.npy').item()
    pdb.set_trace()
    #pool = Pool( processes = 3 )
    #pool.map(gen_data_bv, img_lst)

if __name__ == '__main__':
    main()
