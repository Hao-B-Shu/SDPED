import numpy as np
import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import math

class BDCN_tracing_loss(nn.Module):
    def __init__(self,device='cpu',balance=1.1):
        super(BDCN_tracing_loss, self).__init__()
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
        loss = loss + self.tracingloss(Pred[5], label, tex_factor=0.1, bdr_factor=2.)
        loss = loss + self.tracingloss(Pred[6], label, tex_factor=0.1, bdr_factor=2.)
        loss = loss + self.tracingloss(Pred[7], label, tex_factor=0.1, bdr_factor=2.)
        loss = loss + self.tracingloss(Pred[8], label, tex_factor=0.05, bdr_factor=1)
        loss = loss + self.tracingloss(Pred[9], label, tex_factor=0.05, bdr_factor=1)
        loss = loss + self.tracingloss(Pred[10], label, tex_factor=0.02, bdr_factor=4)
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

def cross_entropy_loss2d(inputs, targets):
    b, c, h, w = inputs.size()
    targets = targets.long()
    total_mask=[]
    for i in range(b):
        mask = targets[i,:,:,:].float()
        num_positive = torch.sum((mask > 0.0).float()).float()  # >0.1
        num_negative = torch.sum((mask <= 0.0).float()).float()  # <= 0.1
        mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative)  # 0.1
        mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
        total_mask.append(mask)
    total_mask=torch.stack(total_mask,dim=0)
    loss = nn.BCELoss(total_mask, size_average=False)(inputs, targets.float())
    return loss/b

def BDCN_loss(out,labels):
    loss = 0
    for k in range(10):
        loss += 0.5 * cross_entropy_loss2d(out[k], labels)
    loss += 1.1 * cross_entropy_loss2d(out[-1], labels)
    return loss

class VGG16_C(nn.Module):
    """"""
    def __init__(self, pretrain=None, logger=None):
        super(VGG16_C, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, (3, 3), stride=1, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=1, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, (3, 3), stride=1, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3_1 = nn.Conv2d(128, 256, (3, 3), stride=1, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, (3, 3), stride=1, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4_1 = nn.Conv2d(256, 512, (3, 3), stride=1, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=1, ceil_mode=True)
        self.conv5_1 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, (3, 3), stride=1, padding=2, dilation=2)
        self.relu5_3 = nn.ReLU(inplace=True)
        if pretrain:
            if '.npy' in pretrain:
                state_dict = np.load(pretrain).item()
                for k in state_dict:
                    state_dict[k] = torch.from_numpy(state_dict[k])
            else:
                state_dict = torch.load(pretrain)
            own_state_dict = self.state_dict()
            for name, param in own_state_dict.items():
                if name in state_dict:
                    if logger:
                        logger.info('copy the weights of %s from pretrained model' % name)
                    param.copy_(state_dict[name])
                else:
                    if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % name)
                    if 'bias' in name:
                        param.zero_()
                    else:
                        param.normal_(0, 0.01)
        else:
            self._initialize_weights(logger)

    def forward(self, x):
        conv1_1 = self.relu1_1(self.conv1_1(x))
        conv1_2 = self.relu1_2(self.conv1_2(conv1_1))
        pool1 = self.pool1(conv1_2)
        conv2_1 = self.relu2_1(self.conv2_1(pool1))
        conv2_2 = self.relu2_2(self.conv2_2(conv2_1))
        pool2 = self.pool2(conv2_2)
        conv3_1 = self.relu3_1(self.conv3_1(pool2))
        conv3_2 = self.relu3_2(self.conv3_2(conv3_1))
        conv3_3 = self.relu3_3(self.conv3_3(conv3_2))
        pool3 = self.pool3(conv3_3)
        conv4_1 = self.relu4_1(self.conv4_1(pool3))
        conv4_2 = self.relu4_2(self.conv4_2(conv4_1))
        conv4_3 = self.relu4_3(self.conv4_3(conv4_2))
        pool4 = self.pool4(conv4_3)
        # pool4 = conv4_3
        conv5_1 = self.relu5_1(self.conv5_1(pool4))
        conv5_2 = self.relu5_2(self.conv5_2(conv5_1))
        conv5_3 = self.relu5_3(self.conv5_3(conv5_2))

        side = [conv1_1, conv1_2, conv2_1, conv2_2,
                conv3_1, conv3_2, conv3_3, conv4_1,
                conv4_2, conv4_3, conv5_1, conv5_2, conv5_3]
        return side

    def _initialize_weights(self, logger=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if logger:
                        logger.info('init the weights of %s from mean 0, std 0.01 gaussian distribution'\
                         % m)
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def crop(data1, data2, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    _, _, h2, w2 = data2.size()
    assert(h2 <= h1 and w2 <= w1)
    data = data1[:, :, crop_h:crop_h+h2, crop_w:crop_w+w2]
    return data

def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

class MSBlock(nn.Module):
    def __init__(self, c_in, rate=4):
        super(MSBlock, self).__init__()
        c_out = c_in
        self.rate = rate

        self.conv = nn.Conv2d(c_in, 32, 3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        dilation = self.rate*1 if self.rate >= 1 else 1
        self.conv1 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu1 = nn.ReLU(inplace=True)
        dilation = self.rate*2 if self.rate >= 1 else 1
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu2 = nn.ReLU(inplace=True)
        dilation = self.rate*3 if self.rate >= 1 else 1
        self.conv3 = nn.Conv2d(32, 32, 3, stride=1, dilation=dilation, padding=dilation)
        self.relu3 = nn.ReLU(inplace=True)

        self._initialize_weights()

    def forward(self, x):
        o = self.relu(self.conv(x))
        o1 = self.relu1(self.conv1(o))
        o2 = self.relu2(self.conv2(o))
        o3 = self.relu3(self.conv3(o))
        out = o + o1 + o2 + o3
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

class BDCN_Edge(nn.Module):
    def __init__(self, pretrain=None, logger=None, rate=4):
        super(BDCN_Edge, self).__init__()
        self.pretrain = pretrain
        t = 1

        self.features = VGG16_C(pretrain, logger)
        self.msblock1_1 = MSBlock(64, rate)
        self.msblock1_2 = MSBlock(64, rate)
        self.conv1_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv1_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn1_1 = nn.Conv2d(21, 1, 1, stride=1)
        self.msblock2_1 = MSBlock(128, rate)
        self.msblock2_2 = MSBlock(128, rate)
        self.conv2_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv2_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn2 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn2_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock3_1 = MSBlock(256, rate)
        self.msblock3_2 = MSBlock(256, rate)
        self.msblock3_3 = MSBlock(256, rate)
        self.conv3_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv3_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn3 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn3_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock4_1 = MSBlock(512, rate)
        self.msblock4_2 = MSBlock(512, rate)
        self.msblock4_3 = MSBlock(512, rate)
        self.conv4_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv4_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn4 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn4_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.msblock5_1 = MSBlock(512, rate)
        self.msblock5_2 = MSBlock(512, rate)
        self.msblock5_3 = MSBlock(512, rate)
        self.conv5_1_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_2_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.conv5_3_down = nn.Conv2d(32*t, 21, (1, 1), stride=1)
        self.score_dsn5 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.score_dsn5_1 = nn.Conv2d(21, 1, (1, 1), stride=1)
        self.upsample_2 = nn.ConvTranspose2d(1, 1, 4, stride=2, bias=False)
        self.upsample_4 = nn.ConvTranspose2d(1, 1, 8, stride=4, bias=False)
        self.upsample_8 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.upsample_8_5 = nn.ConvTranspose2d(1, 1, 16, stride=8, bias=False)
        self.fuse = nn.Conv2d(10, 1, 1, stride=1)

        self._initialize_weights(logger)

    def forward(self, x):
        features = self.features(x)
        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
                self.conv1_2_down(self.msblock1_2(features[1]))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)
        # print(s1.data.shape, s11.data.shape)
        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
            self.conv2_2_down(self.msblock2_2(features[3]))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        # print(s2.data.shape, s21.data.shape)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)
        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
            self.conv3_2_down(self.msblock3_2(features[5])) + \
            self.conv3_3_down(self.msblock3_3(features[6]))
        s3 = self.score_dsn3(sum3)
        s3 =self.upsample_4(s3)
        # print(s3.data.shape)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum3)
        s31 =self.upsample_4(s31)
        # print(s31.data.shape)
        s31 = crop(s31, x, 2, 2)
        sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
            self.conv4_2_down(self.msblock4_2(features[8])) + \
            self.conv4_3_down(self.msblock4_3(features[9]))
        s4 = self.score_dsn4(sum4)
        s4 = self.upsample_8(s4)
        # print(s4.data.shape)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum4)
        s41 = self.upsample_8(s41)
        # print(s41.data.shape)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
            self.conv5_2_down(self.msblock5_2(features[11])) + \
            self.conv5_3_down(self.msblock5_3(features[12]))
        s5 = self.score_dsn5(sum5)
        s5 = self.upsample_8_5(s5)
        # print(s5.data.shape)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum5)
        s51 = self.upsample_8_5(s51)
        # print(s51.data.shape)
        s51 = crop(s51, x, 0, 0)
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51

        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))

        return [F.sigmoid(p1_1), F.sigmoid(p2_1), F.sigmoid(p3_1), F.sigmoid(p4_1), F.sigmoid(p5_1), F.sigmoid(p1_2), F.sigmoid(p2_2), F.sigmoid(p3_2), F.sigmoid(p4_2), F.sigmoid(p5_2), F.sigmoid(fuse)]

    def _initialize_weights(self, logger=None):
        for name, param in self.state_dict().items():
            if self.pretrain and 'features' in name:
                continue
            # elif 'down' in name:
            #     param.zero_()
            elif 'upsample' in name:
                if logger:
                    logger.info('init upsamle layer %s ' % name)
                k = int(name.split('.')[0].split('_')[1])
                param.copy_(get_upsampling_weight(1, 1, k*2))
            elif 'fuse' in name:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    nn.init.constant(param, 0.080)
            else:
                if logger:
                    logger.info('init params %s ' % name)
                if 'bias' in name:
                    param.zero_()
                else:
                    param.normal_(0, 0.01)
        # print self.conv1_1_down.weight

if __name__ == '__main__':
    model = BDCN_Edge()
    a=torch.rand(2,3,100,100)
    a=torch.autograd.Variable(a)
    for x in model(a):
        print(x.data.shape)
    # for name, param in model.state_dict().items():
    #     print name, param
