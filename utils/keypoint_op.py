#-*-coding:utf8-*-
import collections

import cv2
import numpy as np
import torch


def filter_points(points, shape):
    """
    :param points: (N,2), yx format
    :param shape: (H, W)
    :return: filtered point without outliers
    """
    H,W = shape
    if len(points)!=0:
        mask = (points >= 0) & (points <= np.array((H,W))-1)
        mask = np.all(mask, axis=1)
        return points[mask]
    else:
        return points


def compute_keypoint_map(points, shape):
    """
    :param shape: (H,W)
    :param points: (N, 2), yx format
    """
    H,W = shape

    coord = np.minimum(np.round(points).astype(int), np.array((H,W))-1)
    kmap = np.zeros((H,W),dtype=int)
    kmap[coord[:,0],coord[:,1]] = 1
    return kmap


def warp_points(points, homography):
    """
        :param points,[N,2], yx format
        :param homography,[3, 3]
        :return: warped points [N,2], yx format
    """
    if len(points) == 0:
        return points

    points = points.astype(np.float32)

    #To xy format
    points = np.fliplr(points)

    #using opencv
    points = points.reshape(-1, 1, 2)
    warped_points = cv2.perspectiveTransform(points, homography)
    warped_points = warped_points.squeeze(axis=1)#to [N,2]

    #To yx format
    warped_points = np.fliplr(warped_points)
    return warped_points




if __name__=='__main__':
    #testing code
    pts = np.array([[0, 0], [800, 0], [0, 600.], [800., 600.]],dtype=np.float32)#xy format
    wpts = np.array([[30,30.],[800,300],[0,600.],[600., 800.]], dtype=np.float32)#xy format

    h_mat = cv2.getPerspectiveTransform(pts, wpts)

    img = cv2.imread("D:\\cat.jpg")
    H,W,C = img.shape

    vpts = np.fliplr(np.array([[248,253],[346,340],[614,529]],dtype=int))#yx format
    wpts = warp_points(vpts,h_mat)
    wpts = filter_points(wpts,(H,W))
    #to xy format
    vpts = np.fliplr(vpts)
    wpts = np.fliplr(wpts)

    wimg = cv2.warpPerspective(img, h_mat, (W,H))

    cv2.drawMarker(img, vpts[0].astype(int), (0,255,0), 0)
    cv2.drawMarker(img, vpts[1].astype(int), (0, 255, 0), 0)
    cv2.imshow("original", img)

    cv2.drawMarker(wimg, wpts[0].astype(int), (0, 255, 0), 0)
    cv2.drawMarker(wimg, wpts[1].astype(int), (0, 255, 0), 0)
    cv2.imshow("warp", wimg)

    kpmap = compute_keypoint_map(np.fliplr(wpts), (H,W))
    kpmap = (kpmap*255).astype(np.uint8)
    cv2.imshow("kpmap", kpmap)


    cv2.waitKey()
    cv2.destroyAllWindows()

