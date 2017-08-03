import os, pdb, cv2
import numpy as np
import matplotlib.pyplot as plt

def show_data_bv(ID):
    img_path = '/media/disk3/data/scannet'
    ID = ID + '_00'
    data_path = img_path + '/' + ID
    grid_rgb = np.load(data_path + '/grid_rgb_dense.npy')
    grid_depth = np.load(data_path + '/grid_depth_dense.npy')
    grid_rgb = np.transpose(grid_rgb[(2,1,0),:,:],(1,2,0))
    cv2.imwrite(data_path + '/grid_rgb_dense.jpg', grid_rgb)
    cv2.imwrite(data_path + '/grid_depth_dense.jpg', grid_depth)
    '''
    plt.figure(1)
    plt.imshow(grid_rgb)
    plt.title('rgb')
    plt.show()
    pdb.set_trace()

    plt.figure(2)
    plt.imshow(grid_depth)
    plt.title('depth')
    plt.colorbar()
    plt.show()
    '''



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
        show_data_bv(ID)

if __name__ == '__main__':
    main()
