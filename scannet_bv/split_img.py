import numpy as np
import pdb, os, sys
import matplotlib.pyplot as plt
def gen_split_img(all_split, rgb):
    x, y = (rgb.shape[1]+1)/2, (rgb.shape[2]+1)/2
    for i in range(len(all_split)):
        if i == 0 or i == 1 or i == 3:
            new_rgb = np.zeros((3, max(all_split[0])[1]+1, max(all_split[0])[0]+1))
            offset = min(all_split[i])
            #pdb.set_trace()
            for k in all_split[i]:
                new_rgb[0][k[1]-offset[1]][k[0]-offset[0]] = rgb[0][k[1]][k[0]]
                new_rgb[1][k[1]-offset[1]][k[0]-offset[0]] = rgb[1][k[1]][k[0]]
                new_rgb[2][k[1]-offset[1]][k[0]-offset[0]] = rgb[2][k[1]][k[0]]
        elif i == 2:
            new_rgb = np.zeros((3, max(all_split[2])[1]+1, max(all_split[2])[1] - max(all_split[1])[1]+1) )
            offset = min(all_split[i])
            offset_y = max(all_split[1])[1]
            for k in all_split[i]:
                #print k
                new_rgb[0][k[1]-offset_y][k[0]-offset[0]] = rgb[0][k[1]][k[0]]
                new_rgb[1][k[1]-offset_y][k[0]-offset[0]] = rgb[1][k[1]][k[0]]
                new_rgb[2][k[1]-offset_y][k[0]-offset[0]] = rgb[2][k[1]][k[0]]
        elif i == 4:
            new_rgb = np.zeros((3, max(all_split[0])[1]+1, max(all_split[0])[0]+1))
            offset_y = min(all_split[3])[1]
            for k in all_split[i]:
                new_rgb[0][k[1]-offset_y][k[0]] = rgb[0][k[1]][k[0]]
                new_rgb[1][k[1]-offset_y][k[0]] = rgb[1][k[1]][k[0]]
                new_rgb[2][k[1]-offset_y][k[0]] = rgb[2][k[1]][k[0]]
        elif i ==5:
            new_rgb = np.zeros((3, max(all_split[2])[1]+1, max(all_split[2])[1] - max(all_split[1])[1]+1) )
            offset_y = min(all_split[5])[1]
            for k in all_split[i]:
                new_rgb[0][k[1]-offset_y][k[0]] = rgb[0][k[1]][k[0]]
                new_rgb[1][k[1]-offset_y][k[0]] = rgb[1][k[1]][k[0]]
                new_rgb[2][k[1]-offset_y][k[0]] = rgb[2][k[1]][k[0]]
        new_rgb = np.transpose(new_rgb, (1,2,0))
        plt.imshow(new_rgb)
        pdb.set_trace()


def build_axis():
    pi = np.pi
    return np.array([
                    [-np.cos(pi/6), np.sin(pi/6)], 
                    [0,1], 
                    [np.cos(pi/6), np.sin(pi/6)], 
                    [np.cos(pi/6), -np.sin(pi/6)], 
                    [0, -1], 
                    [-np.cos(pi/6), -np.sin(pi/6)], 
                    [-np.cos(pi/6), np.sin(pi/6)], 
                    ])

def inter(ID):
    img_path = '/home/kevin/3D_project/SCANNET/data/all_data'
    ID = ID + '_00'
    data_path = img_path + '/' + ID
    file_lst = os.listdir(data_path)
    axis = build_axis()
    if 'grid_rgb_sparse.npy' in file_lst:
        rgb = np.load(data_path + '/grid_rgb_sparse.npy')
        x = np.arange(0, rgb.shape[2], 1)
        y = np.arange(0, rgb.shape[1], 1)
        #x = np.arange(0, 5, 1)
        #y = np.arange(0, 11, 1)
        xx, yy = np.meshgrid(x, y)
        xx = xx - len(xx[0])/2
        yy = -(yy-np.max(yy)) - len(yy)/2
        #pdb.set_trace()
        all_split = []
        for k in range(len(axis)-1):
            all_idx = []
            a1, a2 = axis[k][0], axis[k][1]
            a3, a4 = axis[k+1][0], axis[k+1][1]
            if a2 < 0:
                a1, a2 = -a1, -a2
            if a4 < 0:
                a3, a4 = -a3, -a4
            print "Line1 Solution is {m}x - {c}y = 0".format(m=a2, c=a1)
            print "Line2 Solution is {m}x - {c}y = 0".format(m=a4, c=a3)
            for i in range(len(yy)):
                eq1 = a2*xx[i] - a1*yy[i]
                eq2 = a4*xx[i] - a3*yy[i]
                if k == 0 or k ==1:
                    idx = np.where((eq1>0)*(eq2<0))[0]
                elif k == 2:
                    idx = np.where((eq1>0)*(eq2>0))[0]
                elif k == 3 or k == 4:
                    idx = np.where((eq1<0)*(eq2>0))[0]
                elif k == 5 :
                    idx = np.where((eq1<0)*(eq2<0))[0]
                xs = xx[i][idx]
                ys = yy[i][idx]
                xs = xs + len(xx[0])/2
                ys = -(ys - len(yy)/2)
                z = zip(xs, ys)
                #print z
                if z != []:
                    all_idx = all_idx + z
            all_split.append(all_idx)
        #pdb.set_trace()
        gen_split_img(all_split, rgb)

def parse_lst(lst):
    new_lst = []
    for ID in lst:
        new_ID = ID.split('_')[0]
        if not new_ID in new_lst:
            new_lst.append(new_ID)
    return new_lst

if __name__ == '__main__':
    data_path = '/media/disk3/data/scannet/'
    data_lst = sorted(os.listdir(data_path))
    data_lst = parse_lst(data_lst)
    for ID in data_lst:
        inter(ID)
        pdb.set_trace()
