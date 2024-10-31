import cv2
import os
import glob
import numpy as np

def initializer(path):
    all_images = sorted(glob.glob(path+os.sep+'*'))
    print('Found {} Images for stitching'.format(len(all_images)))
    imageSet = [cv2.imread(each) for each in all_images]
    images = [cv2.resize(each,(480,320)) for each in imageSet ]
    cylinderical_images = [change_coordinates_to_cylinderical(each) for each in images]
    count = len(images)
    centerIdx = int(count/2)
    return cylinderical_images , centerIdx

def change_coordinates_to_cylinderical(start_image):
    global w, h, center, f
    h, w = start_image.shape[:2]
    center = [w // 2, h // 2]
    f = 600  

    modified_image = np.zeros(start_image.shape, dtype=np.uint8)

    AllCoordinates_of_ti =  np.array([np.array([i, j]) for i in range(w) for j in range(h)])
    ti_x = AllCoordinates_of_ti[:, 0]
    ti_y = AllCoordinates_of_ti[:, 1]
    ii_x, ii_y = Convert_xy(ti_x, ti_y)
    ii_tl_x = ii_x.astype(int)
    ii_tl_y = ii_y.astype(int)

    GoodIndices = (ii_tl_x >= 0) * (ii_tl_x <= (w-2)) * \
                    (ii_tl_y >= 0) * (ii_tl_y <= (h-2))

    ti_x = ti_x[GoodIndices]
    ti_y = ti_y[GoodIndices]
    ii_x = ii_x[GoodIndices]
    ii_y = ii_y[GoodIndices]
    ii_tl_x = ii_tl_x[GoodIndices]
    ii_tl_y = ii_tl_y[GoodIndices]
    dx = ii_x - ii_tl_x
    dy = ii_y - ii_tl_y

    weight_tl = (1.0 - dx) * (1.0 - dy)
    weight_tr = (dx)       * (1.0 - dy)
    weight_bl = (1.0 - dx) * (dy)
    weight_br = (dx)       * (dy)

    modified_image[ti_y, ti_x, :] = ( weight_tl[:, None] * start_image[ii_tl_y,     ii_tl_x,     :] ) + \
                                        ( weight_tr[:, None] * start_image[ii_tl_y,     ii_tl_x + 1, :] ) + \
                                        ( weight_bl[:, None] * start_image[ii_tl_y + 1, ii_tl_x,     :] ) + \
                                        ( weight_br[:, None] * start_image[ii_tl_y + 1, ii_tl_x + 1, :] )

    min_x = min(ti_x)
    modified_image = modified_image[:, min_x : -min_x, :]
    return modified_image

def Convert_xy(x, y):
    global center, f
    xt = ( f * np.tan( (x - center[0]) / f ) ) + center[0]
    yt = ( (y - center[1]) / np.cos( (x - center[0]) / f ) ) + center[1]
    return xt, yt


    
    