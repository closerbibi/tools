def constructing_grid_pj(max_idxy,idxx,idxy,grid,pc,largexy,img_idx):

    ### color image or hha image, both are (2,1,0)
    #hha_name = '/home/closerbibi/workspace/data/hha/NYU%04d.png'%(int(img_idx))
    #img = cv2.imread(hha_name)
    color_name = '/home/closerbibi/workspace/data/NYUimg_only/NYU%04d.jpg'%(int(img_idx))
    img = cv2.imread(color_name)
    disparity = img[:,:,2] # hha: disparity
    height = img[:,:,1] # hha: height
    angle = img[:,:,0] # hha: angle
    for ix in idxx:
        for iy in idxy:
            rviy = np.nanmax(idxy) - iy
            # location of bv
            xy_eligible = (largexy[0].astype(int)==ix)*(largexy[1].astype(int)==iy) # "*" means binary operator "and"~
            xy_eligible = np.where(xy_eligible==True)[0]
            # channel 0: disparity
            # channel 1: height
            # channel 2: angle
            if len(xy_eligible) > 0:
                pdb.set_trace()
                # then find the max z position to complete the map, bv can only see the highest point
                location = xy_eligible[np.argmax(pc[2][xy_eligible])]
                hh = height.shape[0]; ww = height.shape[1];
                wmap = int(np.floor(location/hh))
                hmap = (location%hh)-1
                grid[0][rviy][ix] = disparity[hmap,wmap]
                grid[1][rviy][ix] = height[hmap,wmap]
                grid[2][rviy][ix] = angle[hmap,wmap]
            else:
                grid[0][rviy][ix] = 0 
                grid[1][rviy][ix] = 0 #np.nanmin(height)
                grid[2][rviy][ix] = 0 
                                                                                                                                                                                                     
    pdb.set_trace()
    return grid

