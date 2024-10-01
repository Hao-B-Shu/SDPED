import os, sys
import torch
import torch.nn as nn
import torchvision.models as models
import torch.autograd.variable as Variable
import numpy as np
import scipy.io as sio
from torch.nn.modules.conv import _ConvNd
#from torch.nn.modules.conv import _single, _pair, _triple
import torch.nn.functional as F

class RCF_tracing_loss(nn.Module):
    def __init__(self,device='cpu',balance=1.1):
        super(RCF_tracing_loss, self).__init__()
        self.device=device
        self.mask_radius = 2
        self.radius = 2
        self.balance=balance

        self.filt = torch.ones(1, 1, 2 * self.radius + 1, 2 * self.radius + 1)
        self.filt.requires_grad = False
        self.filt = self.filt.to(self.device)

        self.filt1 = torch.ones(1, 1, 3, 3)
        self.filt1.requires_grad = False
        self.filt1 = self.filt1.to(self.device)
        self.filt2 = torch.ones(1, 1, 2 * self.mask_radius + 1, 2 * self.mask_radius + 1)
        self.filt2.requires_grad = False
        self.filt2 = self.filt2.to(self.device)

    def forward(self, Pred, label):
        loss = 0
        loss = loss + self.tracingloss(Pred[0], label, tex_factor=0.1, bdr_factor=2.)
        loss = loss + self.tracingloss(Pred[1], label, tex_factor=0.1, bdr_factor=2.)
        loss = loss + self.tracingloss(Pred[2], label, tex_factor=0.1, bdr_factor=2.)
        loss = loss + self.tracingloss(Pred[3], label, tex_factor=0.05, bdr_factor=1)
        loss = loss + self.tracingloss(Pred[4], label, tex_factor=0.05, bdr_factor=1)
        loss = loss + self.tracingloss(Pred[5], label, tex_factor=0.02, bdr_factor=4)
        return loss

    def bdrloss(self,prediction, label):
        bdr_pred = prediction * label
        pred_bdr_sum = label * F.conv2d(bdr_pred, self.filt, bias=None, stride=1, padding=self.radius)
        texture_mask = F.conv2d(label.float(), self.filt, bias=None, stride=1, padding=self.radius)
        mask = (texture_mask != 0).float()
        mask[label == 1] = 0
        pred_texture_sum = F.conv2d(prediction * (1-label) * mask, self.filt, bias=None, stride=1, padding=self.radius)
        softmax_map = torch.clamp(pred_bdr_sum / (pred_texture_sum + pred_bdr_sum + 1e-10), 1e-10, 1 - 1e-10)
        cost = -label * torch.log(softmax_map)
        cost[label == 0] = 0
        return cost.sum()

    def textureloss(self,prediction, label):
        pred_sums = F.conv2d(prediction.float(), self.filt1, bias=None, stride=1, padding=1)
        label_sums = F.conv2d(label.float(), self.filt2, bias=None, stride=1, padding=self.mask_radius)
        mask = 1 - torch.gt(label_sums, 0).float()
        loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
        loss[mask == 0] = 0
        return torch.sum(loss)

    def tracingloss(self,prediction, label, tex_factor=0., bdr_factor=0.):
        label = label.float()
        prediction = prediction.float()
        with torch.no_grad():
            mask = label.clone()
            num_positive = torch.sum((mask==1).float()).float()
            num_negative = torch.sum((mask==0).float()).float()
            beta = num_negative / (num_positive + num_negative)
            mask[mask == 1] = beta
            mask[mask == 0] = self.balance * (1 - beta)
            mask[mask == 2] = 0
        cost = torch.sum(torch.nn.functional.binary_cross_entropy(prediction.float(),label.float(), weight=mask, reduce=False))
        label_w = (label != 0).float()
        textcost = self.textureloss(prediction.float(),label_w.float())
        bdrcost = self.bdrloss(prediction.float(),label_w.float())
        return cost + bdr_factor*bdrcost + tex_factor*textcost

def RCF_weight_cross_entropy(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduction='none')
    return torch.sum(cost)

def RCF_loss(preds,label,weight=[1,1,1,1,1,1]):
    loss=0
    for i in range (len(preds)):
        loss=loss+RCF_weight_cross_entropy(preds[i],label)*weight[i]
    return loss

class DilateConv(nn.Module):
    """
    d_rate: dilation rate
    H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)
    set kernel size to 3, stride to 1, padding==d_rate ==> spatial size kept
    """

    def __init__(self, d_rate, in_ch, out_ch):
        super(DilateConv, self).__init__()
        self.d_conv = nn.Conv2d(in_ch, out_ch, kernel_size=3,
                                stride=1, padding=d_rate, dilation=d_rate)

    def forward(self, x):
        return self.d_conv(x)


class RCF_Edge(nn.Module):
    def __init__(self,device='cpu'):
        super(RCF_Edge, self).__init__()

        self.device=device

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        # lr 100 200 decay 1 0
        # self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        # self.conv5_1 = DilateConv(d_rate=2, in_ch=512, out_ch=512) # error ! name conv5_1.dconv.weight erro in load vgg16
        # self.conv5_2 = DilateConv(d_rate=2, in_ch=512, out_ch=512)
        # self.conv5_3 = DilateConv(d_rate=2, in_ch=512, out_ch=512)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3,
                                 stride=1, padding=2, dilation=2)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.maxpool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)

        # lr 0.1 0.2 decay 1 0
        self.conv1_1_down = nn.Conv2d(64, 21, 1, padding=0)
        self.conv1_2_down = nn.Conv2d(64, 21, 1, padding=0)

        self.conv2_1_down = nn.Conv2d(128, 21, 1, padding=0)
        self.conv2_2_down = nn.Conv2d(128, 21, 1, padding=0)

        self.conv3_1_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_2_down = nn.Conv2d(256, 21, 1, padding=0)
        self.conv3_3_down = nn.Conv2d(256, 21, 1, padding=0)

        self.conv4_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv4_3_down = nn.Conv2d(512, 21, 1, padding=0)

        self.conv5_1_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_2_down = nn.Conv2d(512, 21, 1, padding=0)
        self.conv5_3_down = nn.Conv2d(512, 21, 1, padding=0)

        # lr 0.01 0.02 decay 1 0
        self.score_dsn1 = nn.Conv2d(21, 1, 1)
        self.score_dsn2 = nn.Conv2d(21, 1, 1)
        self.score_dsn3 = nn.Conv2d(21, 1, 1)
        self.score_dsn4 = nn.Conv2d(21, 1, 1)
        self.score_dsn5 = nn.Conv2d(21, 1, 1)
        # lr 0.001 0.002 decay 1 0
        self.score_final = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        # VGG
        img_H, img_W = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))
        pool4 = self.maxpool4(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))

        conv1_1_down = self.conv1_1_down(conv1_1)
        conv1_2_down = self.conv1_2_down(conv1_2)
        conv2_1_down = self.conv2_1_down(conv2_1)
        conv2_2_down = self.conv2_2_down(conv2_2)
        conv3_1_down = self.conv3_1_down(conv3_1)
        conv3_2_down = self.conv3_2_down(conv3_2)
        conv3_3_down = self.conv3_3_down(conv3_3)
        conv4_1_down = self.conv4_1_down(conv4_1)
        conv4_2_down = self.conv4_2_down(conv4_2)
        conv4_3_down = self.conv4_3_down(conv4_3)
        conv5_1_down = self.conv5_1_down(conv5_1)
        conv5_2_down = self.conv5_2_down(conv5_2)
        conv5_3_down = self.conv5_3_down(conv5_3)

        so1_out = self.score_dsn1(conv1_1_down + conv1_2_down)
        so2_out = self.score_dsn2(conv2_1_down + conv2_2_down)
        so3_out = self.score_dsn3(conv3_1_down + conv3_2_down + conv3_3_down)
        so4_out = self.score_dsn4(conv4_1_down + conv4_2_down + conv4_3_down)
        so5_out = self.score_dsn5(conv5_1_down + conv5_2_down + conv5_3_down)
        ## transpose and crop way
        weight_deconv2 = make_bilinear_weights(4, 1).to(self.device)
        weight_deconv3 = make_bilinear_weights(8, 1).to(self.device)
        weight_deconv4 = make_bilinear_weights(16, 1).to(self.device)
        weight_deconv5 = make_bilinear_weights(32, 1).to(self.device)

        upsample2 = torch.nn.functional.conv_transpose2d(so2_out, weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(so3_out, weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(so4_out, weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(so5_out, weight_deconv5, stride=8)
        ### center crop
        so1 = crop(so1_out, img_H, img_W)
        so2 = crop(upsample2, img_H, img_W)
        so3 = crop(upsample3, img_H, img_W)
        so4 = crop(upsample4, img_H, img_W)
        so5 = crop(upsample5, img_H, img_W)
        ### crop way suggested by liu
        # so1 = crop_caffe(0, so1, img_H, img_W)
        # so2 = crop_caffe(1, upsample2, img_H, img_W)
        # so3 = crop_caffe(2, upsample3, img_H, img_W)
        # so4 = crop_caffe(4, upsample4, img_H, img_W)
        # so5 = crop_caffe(8, upsample5, img_H, img_W)
        ## upsample way
        # so1 = F.upsample_bilinear(so1, size=(img_H,img_W))
        # so2 = F.upsample_bilinear(so2, size=(img_H,img_W))
        # so3 = F.upsample_bilinear(so3, size=(img_H,img_W))
        # so4 = F.upsample_bilinear(so4, size=(img_H,img_W))
        # so5 = F.upsample_bilinear(so5, size=(img_H,img_W))

        fusecat = torch.cat((so1, so2, so3, so4, so5), dim=1)
        fuse = self.score_final(fusecat)
        results = [so1, so2, so3, so4, so5, fuse]
        results = [torch.sigmoid(r) for r in results]
        return results


def crop(variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


def crop_caffe(location, variable, th, tw):
    h, w = variable.shape[2], variable.shape[3]
    x1 = int(location)
    y1 = int(location)
    return variable[:, :, y1: y1 + th, x1: x1 + tw]


# make a bilinear interpolation kernel
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(in_channels, out_channels, h, w):
    weights = np.zeros([in_channels, out_channels, h, w])
    if in_channels != out_channels:
        raise ValueError("Input Output channel!")
    if h != w:
        raise ValueError("filters need to be square!")
    filt = upsample_filt(h)
    weights[range(in_channels), range(out_channels), :, :] = filt
    return np.float32(weights)


def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w

def upsample(input, stride, num_channels=1):
    kernel_size = stride * 2
    kernel = make_bilinear_weights(kernel_size, num_channels).cuda()
    return torch.nn.functional.conv_transpose2d(input, kernel, stride=stride)


if __name__ == '__main__':
    model = RCF_Edge()
    dummy_input = torch.rand(4, 3, 320, 320)
    output = model(dummy_input)
    for out in output:
        print(out.size())
