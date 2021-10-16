#-*-coding:utf8-*-
import numpy as np
import torch
import cv2
from model.kp_pvt import KPPVT
from utils.plt import make_plot
import yaml
from solver.nms import spatial_nms


def nn_match_two_way(desc1, desc2, nn_thresh=0.6):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches.T

def extract_desc(desc, kpts, H=None, W=None, device='cpu'):
    '''
    :param desc: dict, {'desc_raw':np.ndarray(1CHcWc),'desc':np.ndarray(1CHW)}
    :param kpts:np.ndarray, [[y0,x0],[y1,x1],...]
    :return:descriptors corresponding to each keypoints
    '''
    if len(kpts) !=0:
        kpts_torch = torch.as_tensor(kpts, device=device)
    if desc['desc'] is not None:
        return desc['desc'][:,:,kpts_torch[:,0], kpts_torch[:,1]]
    #for superpoint
    D = desc['desc_raw'].shape[1]
    if len(kpts) == 0:
        desc = np.zeros((D, 0))
    else:
        # Interpolate into descriptor map using 2D point locations.
        kpts_torch = torch.flip(kpts_torch,dims=[1,]).contiguous().float()#yx->xy
        kpts_torch[:, 0] = (kpts_torch[:, 0] / (float(W) / 2.)) - 1.
        kpts_torch[:, 1] = (kpts_torch[:, 1] / (float(H) / 2.)) - 1.
        kpts_torch = kpts_torch.view(1, 1, -1, 2).float()
        desc = torch.nn.functional.grid_sample(desc['desc_raw'], kpts_torch,)
        desc = desc.data.cpu().numpy().reshape(D, -1)
        desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
    return desc


@torch.no_grad()
def do_extraction(model, data, threshold=0.01, device='cpu'):
    H,W = data['img'].shape[2:]
    prob, desc = model(data)
    prob = prob['prob']
    prob[prob < threshold] = 0.0
    prob = spatial_nms(prob, nms_radius=4)
    ## to numpy
    prob = prob.cpu().numpy()
    prob = prob[0]#remove dim B
    ##
    kpts_y, kpts_x = np.where(prob>0.0)
    kpts = np.vstack((kpts_y, kpts_x)).T
    ##
    desc = extract_desc(desc, kpts, H, W, device=device)
    desc = desc.squeeze()#256*N,N is point number
    desc = desc.cpu().numpy()

    return kpts, desc


if __name__=='__main__':
    with open('./config/superpoint_train.yaml', 'r') as fin:
        config = yaml.safe_load(fin)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    img1 = cv2.imread('./data/icl_snippet/250.png',0)
    img1 = cv2.resize(img1, (256,192))
    img2 = cv2.imread('./data/icl_snippet/330.png',0)
    img2 = cv2.resize(img2,(256,192))
    #img2 = np.rot90(img2)
    #
    # img2 = cv2.resize(img2, (180,240))
    # zeros = np.zeros((240,320),dtype=np.uint8)
    # zeros[:,0:img2.shape[1]] = img2
    # img2 = zeros
    #img2 = cv2.flip(img2, 1, dst=None)  # 水平镜像
    #img2 = cv2.flip(img2, 0, dst=None)  # 垂直镜像

    input_img1 = torch.as_tensor(img1[np.newaxis,np.newaxis,:,:], dtype=torch.float32).to(device)
    input_img2 = torch.as_tensor(img2[np.newaxis,np.newaxis,:,:], dtype=torch.float32).to(device)
    input_img1 = input_img1/255.
    input_img2 = input_img2/255.
    mask = torch.ones(1,192,256).to(device)

    input_data1 = {'img': input_img1, 'mask': None}
    input_data2 = {'img': input_img2, 'mask': None}

    model = KPPVT(config['model'], device=device).to(device)
    model.load_state_dict(torch.load('./export/no_nms/init_model_1.5e-2.pth'))
    # model = SuperPointBNNet(config=config['model'], device=device)
    # model.load_state_dict(torch.load('./superpoint_bn.pth'))
    model.to(device).eval()

    kpts1, desc1 = do_extraction(model, input_data1, threshold=0.015, device=device)
    kpts2, desc2 = do_extraction(model, input_data2, threshold=0.015, device=device)

    #desc1 = desc1/np.linalg.norm(desc1,axis=0)#256*n
    #desc2 = desc2/np.linalg.norm(desc2,axis=0)#256*m
    matches = nn_match_two_way(desc1, desc2, nn_thresh=0.7)#0.7 for superpoint
    make_plot(img1, img2, kpts1, kpts2, matches, 'im_pair.png')

    print('Done')
