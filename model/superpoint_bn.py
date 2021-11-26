import torch
import torch.nn as nn
from solver.nms import box_nms
from model.modules.cnn.vgg_backbone import VGGBackbone,VGGBackboneBN
from model.modules.cnn.cnn_heads import DetectorHead, DescriptorHead

class SuperPointBNNet(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, config, input_channel=1, grid_size=8, device='cpu', using_bn=True):
        super(SuperPointBNNet, self).__init__()
        self.nms = config['nms']
        self.det_thresh = config['det_thresh']
        self.topk = config['topk']
        if using_bn:
            self.backbone = VGGBackboneBN(config['backbone']['vgg'], input_channel, device=device)
        else:
            self.backbone = VGGBackbone(config['backbone']['vgg'], input_channel, device=device)
        ##
        self.detector_head = DetectorHead(input_channel=config['det_head']['feat_in_dim'],
                                          grid_size=grid_size, using_bn=using_bn)
        self.descriptor_head = DescriptorHead(input_channel=config['des_head']['feat_in_dim'],
                                              output_channel=config['des_head']['feat_out_dim'],
                                              grid_size=grid_size, using_bn=using_bn)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        if isinstance(x, dict):
            feat_map = self.backbone(x['img'])
        else:
            feat_map = self.backbone(x)
        det_outputs = self.detector_head(feat_map)

        prob = det_outputs['prob']
        if self.nms is not None:
            prob = [box_nms(p.unsqueeze(dim=0),
                            self.nms,
                            min_prob=self.det_thresh,
                            keep_top_k=self.topk).squeeze(dim=0) for p in prob]
            prob = torch.stack(prob)
            det_outputs.setdefault('prob_nms',prob)

        pred = prob[prob>=self.det_thresh]
        det_outputs.setdefault('pred', pred)

        desc_outputs = self.descriptor_head(feat_map)
        return {'det_info':det_outputs, 'desc_info':desc_outputs}


if __name__=='__main__':
    model = SuperPointBNNet()
    model.load_state_dict(torch.load('../superpoint_bn.pth'))
    print('Done')
