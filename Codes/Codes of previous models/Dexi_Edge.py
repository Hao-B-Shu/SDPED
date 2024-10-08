import torch
import torch.nn as nn
import torch.nn.functional as F

def CWBCE_Label(inputs, targets, balance=1.1,l_weight=1.1):
    b,c,h,w=targets.shape
    mask = targets.float().clamp(0,1)
    num_positive = mask.sum()
    num_negative = b*c*h*w-num_positive
    weight_positive = 1.0 * num_negative / (num_positive + num_negative)
    weight_negative = balance * num_positive / (num_positive + num_negative)
    weight=mask*weight_positive+(1-mask)*weight_negative
    cost = torch.nn.BCELoss(weight, reduction='mean')(inputs, targets.float())
    return l_weight * cost

def CWBCE_Pred(inputs, targets, balance=1.1,l_weight=1.1):
    b,c,h,w=targets.shape
    mask = targets.float().clamp(0,1)
    num_positive = mask.sum()
    num_negative = b*c*h*w-num_positive
    weight_positive = 1.0 * num_negative / (num_positive + num_negative)
    weight_negative = balance * num_positive / (num_positive + num_negative)
    weight=inputs*weight_positive+(1-inputs)*weight_negative
    cost = torch.nn.BCELoss(reduction='none')(inputs, targets.float())
    cost=(cost*weight).sum()/(b*c*h*w)
    return l_weight * cost

def SyCWBCE(inputs, targets,w=1):
    cost=(CWBCE_Label(inputs,targets)+w*CWBCE_Pred(inputs,targets))/(1+w)
    return cost

def Dexi_SyCWBCE(inputs, targets, l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3],w=0.1):
    loss=0
    for i in range (len(inputs)):
        loss+=SyCWBCE(inputs[i],targets,w=w)*l_weight[i]
    return loss

def bdcn2_loss(inputs, targets, l_weight = 1.1):
    # bdcn loss with the rcf approach
    targets = targets.long()
    # mask = (targets > 0.1).float()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.0).float()).float() # >0.1
    num_negative = torch.sum((mask <= 0.0).float()).float() # <= 0.1

    mask[mask > 0.] = 1.0 * num_negative / (num_positive + num_negative) #0.1
    mask[mask <= 0.] = 1.1 * num_positive / (num_positive + num_negative)  # before mask[mask <= 0.1]
    # mask[mask == 2] = 0
    # inputs= torch.sigmoid(inputs)
    cost = torch.nn.BCELoss(mask, reduction='none')(inputs, targets.float())
    # cost = torch.mean(cost.float().mean((1, 2, 3))) # before sum
    cost = torch.sum(cost.float().mean((1, 2, 3))) # before sum
    return l_weight*cost

def Dexi_loss_bdcn2(inputs, targets, l_weight = [0.7,0.7,1.1,1.1,0.3,0.3,1.3]):
    loss=0
    for i in range (len(inputs)):
        loss+=bdcn2_loss(inputs[i],targets,l_weight[i])
    return loss

class Tracing_loss_Dexi(nn.Module):

    def __init__(self, device='cpu',radius=4, mask_radius=4,l_weight = [[0.05, 2.], [0.05, 2.], [0.05, 2.], [0.1, 1.], [0.1, 1.], [0.1, 1.],[0.01, 4.]],balanced_w=1.1):
        super(Tracing_loss_Dexi, self).__init__()
        self.radius=radius
        self.mask_radius=mask_radius
        self.l_weight=l_weight
        self.device = device
        self.filt = torch.ones(1, 1, 2 * self.radius + 1, 2 * self.radius + 1)
        self.filt.requires_grad = False
        self.filt = self.filt.to(self.device)
        self.filt1 = torch.ones(1, 1, 3, 3)
        self.filt1.requires_grad = False
        self.filt1 = self.filt1.to(device)
        self.filt2 = torch.ones(1, 1, 2 * self.mask_radius + 1, 2 * self.mask_radius + 1)
        self.filt2.requires_grad = False
        self.filt2 = self.filt2.to(device)
        self.balanced_w = balanced_w

    def forward(self, inputs, targets):
        loss=0
        for i in range (len(inputs)):
            loss+=self.cats_loss(inputs[i],targets,self.l_weight[i])
        return loss

    def bdrloss(self, prediction, label):
        '''
        The boundary tracing loss that handles the confusing pixels.
        '''

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

    def textureloss(self, prediction, label):
        '''
        The texture suppression loss that smooths the texture regions.
        '''

        pred_sums = F.conv2d(prediction.float(), self.filt1, bias=None, stride=1, padding=1)
        label_sums = F.conv2d(label.float(), self.filt2, bias=None, stride=1, padding=self.mask_radius)

        mask = 1 - torch.gt(label_sums, 0).float()

        loss = -torch.log(torch.clamp(1-pred_sums/9, 1e-10, 1-1e-10))
        loss[mask == 0] = 0

        return torch.sum(loss)

    def cats_loss(self, prediction, label, l_weight):
        # tracingLoss
        tex_factor,bdr_factor = l_weight
        label = label.float()
        prediction = prediction.float()
        with torch.no_grad():
            mask = label.clone()

            num_positive = torch.sum((mask == 1).float()).float()
            num_negative = torch.sum((mask == 0).float()).float()
            beta = num_negative / (num_positive + num_negative)
            mask[mask == 1] = beta
            mask[mask == 0] = self.balanced_w * (1 - beta)
            mask[mask == 2] = 0
        # prediction = torch.sigmoid(prediction)
        # print('bce')
        cost = torch.sum(torch.nn.functional.binary_cross_entropy(
            prediction.float(), label.float(), weight=mask, reduce=False))
        label_w = (label != 0).float()
        # print('tex')
        textcost = self.textureloss(prediction.float(), label_w.float())
        bdrcost = self.bdrloss(prediction.float(), label_w.float())

        return cost + bdr_factor * bdrcost + tex_factor * textcost


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        # torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        # torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CoFusion(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, out_ch, kernel_size=3,
                               stride=1, padding=1)
        self.relu = nn.ReLU()

        self.norm_layer1 = nn.GroupNorm(4, 64)
        self.norm_layer2 = nn.GroupNorm(4, 64)

    def forward(self, x):
        # fusecat = torch.cat(x, dim=1)
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = self.relu(self.norm_layer2(self.conv2(attn)))
        attn = F.softmax(self.conv3(attn), dim=1)

        # return ((fusecat * attn).sum(1)).unsqueeze(1)
        return ((x * attn).sum(1)).unsqueeze(1)

class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        # self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x

        new_features = super(_DenseLayer, self).forward(F.relu(x1))  # F.relu()
        # if new_features.shape[-1]!=x2.shape[-1]:
        #     new_features =F.interpolate(new_features,size=(x2.shape[2],x2.shape[-1]), mode='bicubic',
        #                                 align_corners=False)
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads=[0,0,1,3,7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]  # kernel_size-1
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride,
                 use_bs=True
                 ):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x


class Dexi_Edge(nn.Module):
    """ Definition of the DXtrem network. """

    def __init__(self):
        super(Dexi_Edge, self).__init__()
        self.block_1 = DoubleConvBlock(3, 32, 64, stride=2,)
        self.block_2 = DoubleConvBlock(64, 128, use_act=False)
        self.dblock_3 = _DenseBlock(2, 128, 256) # [128,256,100,100]
        self.dblock_4 = _DenseBlock(3, 256, 512)
        self.dblock_5 = _DenseBlock(3, 512, 512)
        self.dblock_6 = _DenseBlock(3, 512, 256)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections, figure in Journal
        self.side_1 = SingleConvBlock(64, 128, 2)
        self.side_2 = SingleConvBlock(128, 256, 2)
        self.side_3 = SingleConvBlock(256, 512, 2)
        self.side_4 = SingleConvBlock(512, 512, 1)
        self.side_5 = SingleConvBlock(512, 256, 1) # Sory I forget to comment this line :(

        # right skip connections, figure in Journal paper
        self.pre_dense_2 = SingleConvBlock(128, 256, 2)
        self.pre_dense_3 = SingleConvBlock(128, 256, 1)
        self.pre_dense_4 = SingleConvBlock(256, 512, 1)
        self.pre_dense_5 = SingleConvBlock(512, 512, 1)
        self.pre_dense_6 = SingleConvBlock(512, 256, 1)


        self.up_block_1 = UpConvBlock(64, 1)
        self.up_block_2 = UpConvBlock(128, 1)
        self.up_block_3 = UpConvBlock(256, 2)
        self.up_block_4 = UpConvBlock(512, 3)
        self.up_block_5 = UpConvBlock(512, 4)
        self.up_block_6 = UpConvBlock(256, 4)
        self.block_cat = SingleConvBlock(6, 1, stride=1, use_bs=False) # hed fusion method
        # self.block_cat = CoFusion(6,6)# cats fusion method


        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1]!=slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic',align_corners=False)
        else:
            new_tensor=tensor
        # tensor[..., :height, :width]
        return new_tensor

    def forward(self, x):
        assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3) # [128,256,50,50]
        block_3_add = block_3_down + block_2_side
        block_3_side = self.side_3(block_3_add)

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down+block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])
        block_4_down = self.maxpool(block_4)
        block_4_add = block_4_down + block_3_side
        block_4_side = self.side_4(block_4_add)

        # Block 5
        block_5_pre_dense = self.pre_dense_5(
            block_4_down) #block_5_pre_dense_512 +block_4_down
        block_5, _ = self.dblock_5([block_4_add, block_5_pre_dense])
        block_5_add = block_5 + block_4_side

        # Block 6
        block_6_pre_dense = self.pre_dense_6(block_5)
        block_6, _ = self.dblock_6([block_5_add, block_6_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        out_5 = self.up_block_5(block_5)
        out_6 = self.up_block_6(block_6)
        results = [out_1, out_2, out_3, out_4, out_5, out_6]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)  # Bx6xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW

        # return results
        results.append(block_cat)

        results=[F.sigmoid(i) for i in results]
        return results

if __name__ == '__main__':
    batch_size = 8
    img_height = 352
    img_width = 352

    device = "cpu"
    input = torch.rand(batch_size, 3, img_height, img_width).to(device)
    model = Dexi_Edge().to(device)
    output = model(input)
    print(f"output shapes: {[t.shape for t in output]}")
