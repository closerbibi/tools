import numpy as np
import cv2, os, sys, pdb, time
from plyfile import PlyData as plyd
from plyfile import PlyElement as plye
from multiprocessing import Pool
import matplotlib.pyplot as plt
sys.path.append('/home/kevin/3D_project/code')
import basic as ba

def constructing_grid_pj(idxx, idxy, lpc_id, pc, rgb):
    grid_rgb = np.zeros((3, len(idxy),len(idxx)))
    grid_depth = np.zeros((len(idxy),len(idxx)))
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
                #pdb.set_trace()
                # then find the max z position to complete the map, bv can only see the highest point
                original_idx = xy_eligible[np.argmax(lpc_x_sort[2,x_loc][0,xy_eligible])]
                location = lpc_x_sort[3,x_loc][0, original_idx].astype(int) # mapping two times
                #location = xy_eligible[np.argmax(pc[2][xy_eligible])]
                grid_depth[rviy][ix] = max_height - pc[2][location]#depth
                grid_rgb[0][rviy][ix] = rgb[location][0]#R
                grid_rgb[1][rviy][ix] = rgb[location][1]#G
                grid_rgb[2][rviy][ix] = rgb[location][2]#B
            else:
                grid_depth[rviy][ix] = 0
                grid_rgb[0][rviy][ix] = 0
                grid_rgb[1][rviy][ix] = 0 #np.nanmin(height)
                grid_rgb[2][rviy][ix] = 0
    #pdb.set_trace()
    return grid_rgb, grid_depth

def axis_2d(axis):
    return (np.floor(np.array(axis)*100)).tolist()

def gen_box(f_seg, f_par, pc):
    pc[0] = pc[0]-np.nanmin(pc[0])
    pc[1] = pc[1]-np.nanmin(pc[1])
    pc[2] = pc[2]-np.nanmin(pc[2])
    segInd = np.array(f_seg['segIndices'])
    label_group = f_par['segGroups']
    #pdb.set_trace()
    object_lst = {}
    max_y = np.max(pc[1])
    for i in range(len(label_group)):
        if not label_group[i]['label'] in object_lst.keys():
            object_lst[label_group[i]['label']] = []
        mask = np.in1d(segInd, label_group[i]['segments'])
        xyz = []
        for k in range(len(pc)):
            box = pc[k][mask]
            pdb.set_trace()
            if k == 1:
                y_min = max_y - float(np.max(box))
                y_max = max_y - float(np.min(box))
                axis = [y_min, y_max]
            else:
                axis = [float(np.min(box)), float(np.max(box))]
            if k != 2:
                axis = axis_2d(axis)
            xyz.append(axis)
        #pdb.set_trace()
        object_lst[label_group[i]['label']].append(xyz)
    #pdb.set_trace()
    return object_lst

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

def gen_pc_rgb(data):
    x, y, z, rgb = [], [], [], []
    for i in range(len(data)):
        x.append(data[i][0])
        y.append(data[i][1])
        z.append(data[i][2])
        rgb.append([data[i][3], data[i][4], data[i][5]])
    x, y, z = np.array(x), np.array(y), np.array(z)
    rgb = np.array(rgb)
    pc = np.vstack((x, y, z))
    return pc, rgb
    #pdb.set_trace()

def gen_data_bv(ID):
    img_path = '/media/disk3/data/scannet'
    ID = ID + '_00'
    data_path = img_path + '/' + ID
    file_lst = os.listdir(data_path)
    if not 'dense.npy' in file_lst:
        print 'Start at %s !'%ID
        #f = plyd.read(data_path + '/%s_vh_clean.ply'%(ID))
        f = plyd.read(data_path + '/%s_vh_clean.ply'%(ID))
        f_seg = ba.LoadJson(data_path + '/%s_vh_clean_2.0.010000.segs.json'%(ID))
        f_par = ba.LoadJson(data_path + '/%s.aggregation.json'%(ID))
        data = f.elements[0].data
        start = time.time()
        pc, rgb = gen_pc_rgb(data)
        idxx,idxy,largexy = rescale_pc(pc)
        grid_rgb, grid_depth = constructing_grid_pj(idxx, idxy, largexy, pc, rgb)
        total = (time.time() - start)/60
        print '%s %f finish'%(ID, total) 
        #pdb.set_trace()
        bbox = gen_box(f_seg, f_par, pc)
        #pdb.set_trace()
        ba.WriteJson(data_path, 'bbox_dense', bbox)
        np.save(data_path + '/grid_rgb_dense.npy', grid_rgb)
        np.save(data_path + '/grid_depth_dense.npy', grid_depth)
    else:
        print '%s Finish!'%ID


    #pdb.set_trace()

def parse_lst(lst):
    new_lst = []
    for ID in lst:
        new_ID = ID.split('_')[0]
        if not new_ID in new_lst:
            new_lst.append(new_ID)
    return new_lst

def main():
    img_path = '/media/disk3/data/scannet'
    img_lst = sorted(os.listdir(img_path))
    img_lst = parse_lst(img_lst)
    for ID in img_lst:
        gen_data_bv(ID)
    #pdb.set_trace()
    #pool = Pool( processes = 2 )
    #pool.map(gen_data_bv, img_lst)
    pdb.set_trace()

if __name__ == '__main__':
    main()
