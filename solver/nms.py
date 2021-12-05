#-*-coding:utf-8-*-
import torch
import torchvision


def spatial_nms(scores, nms_radius=4, iter_n=0):
    """
    Fast Non-maximum suppression to remove nearby points
    scores: B,H,W
    """
    assert(nms_radius >= 0)
    assert(len(scores.shape)==3)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius*2+1, stride=1, padding=nms_radius)

    scores = scores.unsqueeze(dim=1)
    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(iter_n):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    res = torch.where(max_mask, scores, zeros)
    return res.squeeze(dim=1)


def box_nms(prob, size=4, iou=0.1, min_prob=0.015, keep_top_k=-1):
    """
    :param prob: probability, torch.tensor, must be [1,H,W]
    :param size: box size for 2d nms
    :param iou:
    :param min_prob:
    :param keep_top_k:
    :return:
    """
    assert(prob.shape[0]==1 and len(prob.shape)==3)
    prob = prob.squeeze(dim=0)

    pts = torch.stack(torch.where(prob>=min_prob)).t()
    boxes = torch.cat((pts-size/2.0, pts+size/2.0),dim=1).to(torch.float64)
    scores = prob[pts[:,0],pts[:,1]]
    indices = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou)
    pts = pts[indices,:]
    scores = scores[indices]
    if keep_top_k>0:
        k = min(scores.shape[0], keep_top_k)
        scores, indices = torch.topk(scores,k)
        pts = pts[indices,:]
    nms_prob = torch.zeros_like(prob)
    nms_prob[pts[:,0],pts[:,1]] = scores

    return nms_prob.unsqueeze(dim=0)

if __name__=='__main__':
    pass