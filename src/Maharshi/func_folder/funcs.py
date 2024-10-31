import numpy as np
import cv2
import random
import math

def Image_stitcher(cylinderical_images,centerIdx):
    a = cylinderical_images[centerIdx]
    hom = []
    for b in cylinderical_images[centerIdx+1:]:
        a , homo = wrap(a,b)
        if homo is None:
            break
        hom = np.append(hom,homo)
    right_group = a
    a = cylinderical_images[centerIdx]
    for b in cylinderical_images[0:centerIdx][::-1]:
        a, homo= wrap(a,b)
        if homo is None:
            break
        hom = np.append(hom,homo)
    left_group = a
    a = right_group
    b = left_group
    result,homo = wrap(a,b)
    if homo is not None:
        hom = np.append(hom,homo)
    hom = np.array(hom)
    hom = hom.reshape((-1,3,3))
    return result,hom

def match(i1,i2,direction = None):
    imageSet1 = Sift_out(i1)
    imageSet2 = Sift_out(i2)
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm =0, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(
        imageSet2['des'],
        imageSet1['des'],
        k =2)
    matched_features =[]
    for i,(m,n) in enumerate(matches):
        if(m.distance) < (0.5*n.distance):
            matched_features.append((m.trainIdx,m.queryIdx))
    if len(matched_features) >=10:
        pointsCurrent = imageSet2['kp']
        pointsPrevious = imageSet1['kp']

        matchedPointsCurrent = np.float32(
            [pointsCurrent[i].pt for (_,i) in matched_features]
            )
        matchedPointsPrev = np.float32(
            [pointsPrevious[i].pt for (i,_) in matched_features]
            )
        H,inliers_curr,inliers_prev = ransac(matchedPointsCurrent,matchedPointsPrev,4)
        H = hom_calc(inliers_curr,inliers_prev)
        return H
    return None

def hom_calc(current,previous):
    a_vals = []
    for i in range(len(current)):
        p1 = np.matrix([current[i][0],current[i][1],1])
        p2 = np.matrix([previous[i][0],previous[i][1], 1])

        a2 = [0, 0, 0, -p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2),
                p2.item(1) * p1.item(0), p2.item(1) * p1.item(1), p2.item(1) * p1.item(2)]
        a1 = [-p2.item(2) * p1.item(0), -p2.item(2) * p1.item(1), -p2.item(2) * p1.item(2), 0, 0, 0,
                p2.item(0) * p1.item(0), p2.item(0) * p1.item(1), p2.item(0) * p1.item(2)]
        a_vals.append(a1)
        a_vals.append(a2)

    matrixA = np.matrix(a_vals)
    u, s, v = np.linalg.svd(matrixA)
    H = np.reshape(v[8], (3, 3))
    H = (1/H.item(8)) * H
    return H

def ransac(current,previous, thresh):
    maxInliers_curr, maxInliers_prev =[],[]
    finalH = None
    random.seed(2)
    for i in range(1000):
        currFour = np.empty((0, 2))
        preFour = np.empty((0,2))
        for j in range(4):
            random_pt = random.randrange(0, len(current))
            curr = current[random_pt]
            pre = previous[random_pt]
            currFour = np.vstack((currFour,curr))
            preFour = np.vstack((preFour,pre))


        #call the homography function on those points
        h = hom_calc(currFour,preFour)
        inliers_curr = []
        inliers_prev =[]
        for i in range(len(current)):
            d = Distance(current[i],previous[i], h)
            if d < 10:
                inliers_curr.append([current[i][0],current[i][1]])
                inliers_prev.append([previous[i][0],previous[i][1]])

        if len(inliers_curr) > len(maxInliers_curr):
            maxInliers_curr = inliers_curr
            maxInliers_prev = inliers_prev
            finalH = h

        if len(maxInliers_curr) > (len(current)*thresh):
            break

    return finalH, maxInliers_curr,maxInliers_prev


def Distance(current,previous, h):

    p1 = np.transpose(np.matrix([current[0], current[1], 1]))
    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([previous[0], previous[1], 1]))
    error = p2 - estimatep2
    return np.linalg.norm(error)

def Sift_out(im):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp,des = sift.detectAndCompute(gray,None)
    return {'kp':kp,'des':des}

def remove_extra_pix( image):
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    coords = cv2.findNonZero(gray)  # Returns all non-zero points
    x, y, w, h = cv2.boundingRect(coords) 
    cropped_image = image[y:y+h, x:x+w]
    cv2.imwrite("Cropped_Image.jpg", cropped_image)
    return cropped_image

def homogeneous_coordinate(coordinate):
    x = coordinate[0]/coordinate[2]
    y = coordinate[1]/coordinate[2]
    return x, y

def wrap(a,b):
    H = match(a,b,"left")
    if H is None:
        return a , H
    h1, w1 = b.shape[:2]
    h2, w2 = a.shape[:2]

    row_number, column_number = int(b.shape[0]), int(b.shape[1])
    homography = H
    up_left_cor = homogeneous_coordinate(np.dot(homography, [[0],[0],[1]]))
    up_right_cor = homogeneous_coordinate(np.dot(homography, [[column_number-1],[0],[1]]))
    low_left_cor = homogeneous_coordinate(np.dot(homography, [[0],[row_number-1],[1]]))
    low_right_cor = homogeneous_coordinate(np.dot(homography, [[column_number-1],[row_number-1],[1]]))
    corners2 =np.float32([up_left_cor,low_left_cor,low_right_cor,up_right_cor]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners2, np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)))
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    Hnew = Ht.dot(H)
    homography = Hnew

    offset_x = math.floor(xmin)
    offset_y = math.floor(ymin)

    max_x = math.ceil(xmax)
    max_y = math.ceil(ymax)

    size_x = max_x - offset_x
    size_y = max_y - offset_y

    dsize = [size_x,size_y]
    homography_inverse = np.linalg.inv(homography)

    tmp = np.zeros((dsize[1], dsize[0], 3))
    tmp1= np.zeros((dsize[1],dsize[0],3))

    for x in range(size_x):
        for y in range(size_y):
            point_xy = homogeneous_coordinate(np.dot(homography_inverse, [[x], [y], [1]]))
            point_x = int(point_xy[0])
            point_y = int(point_xy[1])

            if (point_x >= 0 and point_x < column_number and point_y >= 0 and point_y < row_number):
                tmp[y, x, :] = b[point_y, point_x, :]

    tmp1[t[1]:h2+t[1], t[0]:w2+t[0]] = a
    tmp = np.where(np.all(tmp == 0, axis=-1, keepdims=True), tmp1, tmp)
    tmp1 = np.where(np.all(tmp1 == 0, axis=-1, keepdims=True), tmp, tmp1)
    alpha = 0.5
    img_final = cv2.addWeighted(tmp1, alpha, tmp, 1 - alpha, 0)
    img_final = remove_extra_pix(img_final)
    img_final = img_final.astype(a.dtype)
    return img_final , H

    

    


    