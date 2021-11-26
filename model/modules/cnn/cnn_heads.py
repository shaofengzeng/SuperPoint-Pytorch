# -*-coding:utf8-*-
import torch
from utils.tensor_op import pixel_shuffle
import torch.nn.functional as F


class DetectorHead(torch.nn.Module):
    def __init__(self, input_channel, grid_size, using_bn=True):
        super(DetectorHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn
        ##
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)

        self.bnPa,self.bnPb = None,None
        if using_bn:
            self.bnPa = torch.nn.BatchNorm2d(256)
            self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnPa(self.relu(self.convPa(x)))
            out = self.bnPb(self.convPb(out))  #(B,65,H,W)
        else:
            out = self.relu(self.convPa(x))
            out = self.convPb(out)  # (B,65,H,W)

        prob = self.softmax(out)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]

        return {'logits':out, 'prob':prob}


class DescriptorHead(torch.nn.Module):
    def __init__(self, input_channel, output_channel, grid_size, using_bn=True):
        super(DescriptorHead, self).__init__()
        self.grid_size = grid_size
        self.using_bn = using_bn

        self.convDa = torch.nn.Conv2d(input_channel, 256, kernel_size=3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convDb = torch.nn.Conv2d(256, output_channel, kernel_size=1, stride=1, padding=0)

        self.bnDa, self.bnDb = None, None
        if using_bn:
            self.bnDa = torch.nn.BatchNorm2d(256)
            self.bnDb = torch.nn.BatchNorm2d(output_channel)

    def forward(self, x):
        out = None
        if self.using_bn:
            out = self.bnDa(self.relu(self.convDa(x)))
            out = self.bnDb(self.convDb(out))
        else:
            out = self.relu(self.convDa(x))
            out = self.convDb(out)

        # out_norm = torch.norm(out, p=2, dim=1)  # Compute the norm.
        # out = out.div(torch.unsqueeze(out_norm, 1))  # Divide by norm to normalize.

        # TODO: here is different with tf.image.resize_bilinear
        desc = F.interpolate(out, scale_factor=self.grid_size, mode='bilinear',align_corners=False)
        desc = F.normalize(desc, p=2, dim=1)  # normalize by channel

        return {'desc_raw': out, 'desc': desc}
