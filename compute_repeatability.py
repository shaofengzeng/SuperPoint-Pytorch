import cv2
import numpy as np
import os
import solver.detector_evaluation as ev
from utils.plt import plot_imgs

def get_true_keypoints(exper_name, prob_thresh=0.5):
    def warp_keypoints(keypoints, H):
        warped_col0 = np.add(np.sum(np.multiply(keypoints, H[0, :2]), axis=1), H[0, 2])
        warped_col1 = np.add(np.sum(np.multiply(keypoints, H[1, :2]), axis=1), H[1, 2])
        warped_col2 = np.add(np.sum(np.multiply(keypoints, H[2, :2]), axis=1), H[2, 2])
        warped_col0 = np.divide(warped_col0, warped_col2)
        warped_col1 = np.divide(warped_col1, warped_col2)
        new_keypoints = np.concatenate([warped_col0[:, None], warped_col1[:, None]],
                                       axis=1)
        return new_keypoints

    def filter_keypoints(points, shape):
        """ Keep only the points whose coordinates are
        inside the dimensions of shape. """
        mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) & \
               (points[:, 1] >= 0) & (points[:, 1] < shape[1])
        return points[mask, :]

    true_keypoints = []
    for i in range(5):
        path = os.path.join(exper_name, str(i) + ".npz")
        data = np.load(path)
        shape = data['warped_prob'].shape

        # Filter out predictions
        keypoints = np.where(data['prob'] > prob_thresh)
        keypoints = np.stack([keypoints[0], keypoints[1]], axis=-1)
        warped_keypoints = np.where(data['warped_prob'] > prob_thresh)
        warped_keypoints = np.stack([warped_keypoints[0], warped_keypoints[1]], axis=-1)

        # Warp the original keypoints with the true homography
        H = data['homography']
        true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
        true_warped_keypoints[:, [0, 1]] = true_warped_keypoints[:, [1, 0]]
        true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)
        true_keypoints.append((true_warped_keypoints[:, 0], true_warped_keypoints[:, 1]))

    return true_keypoints


def draw_keypoints(img, corners, color=(0, 255, 0), radius=3, s=3):
    img = np.repeat(cv2.resize(img, None, fx=s, fy=s)[..., np.newaxis], 3, -1)
    for c in np.stack(corners).T:
        cv2.circle(img, tuple(s*np.flip(c, 0)), radius, color, thickness=-1)
    return img
def select_top_k(prob, thresh=0, num=300):
    pts = np.where(prob > thresh)
    idx = np.argsort(prob[pts])[::-1][:num]
    pts = (pts[0][idx], pts[1][idx])
    return pts

if __name__=='__main__':

    experiments = ['./data/repeatibility/hpatches/sp_coco_tf_1']
    confidence_thresholds = [0.015, ]

    ## show keypoints
    for i in range(4):
        for e, thresh in zip(experiments, confidence_thresholds):
            path = os.path.join(e, str(i) + ".npz")
            d = np.load(path)
            img = np.round(d['img']*255).astype(np.int).astype(np.uint8)
            warp_img = np.round(d['warp_img']*255).astype(np.int).astype(np.uint8)

            points1 = select_top_k(d['prob'], thresh=thresh)
            im1 = draw_keypoints(img, points1, (0, 255, 0))/255.

            points2 = select_top_k(d['warp_prob'], thresh=thresh)
            im2 = draw_keypoints(warp_img, points2, (0, 255, 0))/255.

            plot_imgs([im1, im2], ylabel=e, dpi=200, cmap='gray',
                      titles=[str(len(points1[0])) + ' points', str(len(points2[0])) + ' points'])

    ## compute repeatability
    for exp, thresh in zip(experiments, confidence_thresholds):
        repeatability = ev.compute_repeatability(exp, keep_k_points=300, distance_thresh=3)
        print('> {}: {}'.format(exp, repeatability))

    # true_keypoints = get_true_keypoints('superpoint_hpatches_repeatability', 0.015)
    # for i in range(3):
    #     e = 'superpoint_hpatches_repeatability'
    #     thresh = 0.015
    #     path = os.path.join("./", e, str(i) + ".npz")
    #     d = np.load(path)
    #
    #     points1 = np.where(d['prob'] > thresh)
    #     im1 = draw_keypoints(d['image'][..., 0] * 255, points1, (0, 255, 0)) / 255.
    #
    #     points2 = true_keypoints[i]
    #     im2 = draw_keypoints(d['warped_image'][..., 0] * 255, points2, (0, 255, 0)) / 255.
    #
    #     plot_imgs([im1, im2], titles=['Original', 'Original points warped'], dpi=200, cmap='gray')