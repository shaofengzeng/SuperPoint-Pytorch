import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_imgs(imgs, titles=None, cmap='brg', ylabel='', normalize=False, ax=None, dpi=100):
    n = len(imgs)
    if not isinstance(cmap, list):
        cmap = [cmap]*n
    if ax is None:
        _, ax = plt.subplots(1, n, figsize=(6*n, 6), dpi=dpi)
        if n == 1:
            ax = [ax]
    else:
        if not isinstance(ax, list):
            ax = [ax]
        assert len(ax) == len(imgs)
    for i in range(n):
        if imgs[i].shape[-1] == 3:
            imgs[i] = imgs[i][..., ::-1]  # BGR to RGB
        ax[i].imshow(imgs[i], cmap=plt.get_cmap(cmap[i]),
                     vmin=None if normalize else 0,
                     vmax=None if normalize else 1)
        if titles:
            ax[i].set_title(titles[i])
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    ax[0].set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()




def make_plot(image0, image1, kpts0, kpts1, matches, im_name):
    """
    :param image0: cv2 image
    :param image1: cv2 image
    :param kpts0: N*2
    :param kpts1: M*2
    :param matches: N*M
    :param im_name: save image name
    :return: None
    """
    margin = 10
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255 * np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0 + margin:] = image1
    out = np.stack([out] * 3, -1)

    ## show_keypoints:
    kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
    white = (255, 255, 255)
    black = (0, 0, 0)
    for y, x in kpts0:
        cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
    for y, x in kpts1:
        cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                   lineType=cv2.LINE_AA)
        cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                   lineType=cv2.LINE_AA)

    # if len(matches)>50:
    #     keep = np.random.choice(len(matches), 50, replace=False)
    #     matches = matches[keep,:]
    for ind0, ind1, score in matches:
        y0,x0 = kpts0[int(ind0),:]
        y1,x1 = kpts1[int(ind1),:]
        ##
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=(235,206,135), thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, color=(235,206,135), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, color=(235,206,135), thickness=-1,
                   lineType=cv2.LINE_AA)
    plt.imshow(out)
    plt.show()
    #cv2.imwrite(im_name, out)