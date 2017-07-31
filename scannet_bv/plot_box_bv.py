import numpy as np
import cv2, os, sys, pdb
import matplotlib.pyplot as plt
sys.path.append('/home/kevin/3D_project/code')
import basic as ba

def swap_cols(arr, frm, to):
    arr[:,[frm, to]] = arr[:,[to, frm]]
    return arr

def plot_2d(data_path, ID):
    #classes = ['toilet', 'chair', 'table', 'bed', 'sofa']
    classes = ['guitar']
    grid_rgb = np.load(data_path + '/grid_rgb_dense.npy')
    #grid_rgb_1 = swap_cols(grid_rgb, 1, 2)
    #grid_rgb = grid_rgb.astype(int)
    #grid_rgb = np.fliplr(grid_rgb.reshape(-1,3)).reshape(grid_rgb.shape)
    grid_rgb_1 = np.transpose(grid_rgb, (1, 2, 0))
    fig,ax = plt.subplots(1)
    plt.imshow(grid_rgb_1)
    bbox = ba.LoadJson(data_path + '/bbox_dense.json')
    #pdb.set_trace()
    for obj in classes:
        for i in range(len(bbox[obj])):
            #print i
            x1x2, y1y2 = bbox[obj][i][0], bbox[obj][i][1]
            ax.add_patch(
                plt.Rectangle((x1x2[0], y1y2[0]),
                                x1x2[1] - x1x2[0], 
                                y1y2[1] - y1y2[0], 
                                fill=False,edgecolor='red', linewidth=2))
    pdb.set_trace()

def parse_lst(lst):
    new_lst = []
    for ID in lst:
        new_ID = ID.split('_')[0]
        if not new_ID in new_lst:
            new_lst.append(new_ID)
    return new_lst

def main():
    img_path = '/home/kevin/3D_project/SCANNET/data/all_data'
    img_lst = sorted(os.listdir(img_path))
    img_lst = parse_lst(img_lst)
    for ID in img_lst[0:1]:
        ID = ID + '_00'
        data_path = img_path + '/' + ID
        file_lst = os.listdir(data_path)
        #pdb.set_trace()
        if 'bbox.json' in file_lst:
            print 'Start!'
            plot_2d(data_path, ID)
        else:
            print 'No bbox in %s'%ID
    #pdb.set_trace()

if __name__ == '__main__':
    main()



