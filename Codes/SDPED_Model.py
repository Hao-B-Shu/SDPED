import os
import torch.nn as nn
import torch.nn.functional as F
import torch


##############loss functions########################
def WBCE(inputs, targets, l_weight=1.1):
    targets = targets.long()
    mask = targets.float()
    num_positive = torch.sum((mask > 0.5).float()).float()
    num_negative = torch.sum((mask <= 0.5).float()).float()
    mask[mask > 0.5] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask <= 0.5] = 1.1 * num_positive / (num_positive + num_negative)
    cost = torch.nn.BCELoss(mask, reduction='mean')(inputs, targets.float())
    return l_weight * cost

def CWBCE(inputs, targets, balance=1.1,l_weight=1.1):
    b,c,h,w=targets.shape
    mask = targets.float().clamp(0,1)
    num_positive = mask.sum()
    num_negative = b*c*h*w-num_positive
    weight_positive = 1.0 * num_negative / (num_positive + num_negative)
    weight_negative = balance * num_positive / (num_positive + num_negative)
    weight=mask*weight_positive+(1-mask)*weight_negative
    cost = torch.nn.BCELoss(weight, reduction='mean')(inputs, targets.float())
    return l_weight * cost
##################################################


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)

def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


##################################Blocks#############################
class SDB(nn.Module):

    def __init__(self, num_feat=64, num_grow_ch=32,trade_SDB=1):
        super(SDB, self).__init__()

        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.trade_SDB=trade_SDB

    def forward(self, x,mid):
        x1 = (self.lrelu(self.conv1(x))+self.trade_SDB*mid[0])/(1+self.trade_SDB)
        x2 = (self.lrelu(self.conv2(torch.cat((x, x1), 1)))+self.trade_SDB*mid[1])/(1+self.trade_SDB)
        x3 = (self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))+self.trade_SDB*mid[2])/(1+self.trade_SDB)
        x4 = (self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))+self.trade_SDB*mid[3])/(1+self.trade_SDB)
        x5 = (self.conv5(torch.cat((x, x1, x2, x3, x4), 1))+ self.trade_SDB*x)/(1+self.trade_SDB)
        return x5,[x1,x2,x3,x4]

class CSDB(nn.Module):

    def __init__(self, num_feat, num_grow_ch,trade_SDB=1):
        super(CSDB, self).__init__()
        self.SRDB1 = SDB(num_feat, num_grow_ch,trade_SDB=trade_SDB)
        self.SRDB2 = SDB(num_feat, num_grow_ch,trade_SDB=trade_SDB)
        self.SRDB3 = SDB(num_feat, num_grow_ch,trade_SDB=trade_SDB)
        self.mid = [0,0,0,0]

        self.num_grow_ch=num_grow_ch

    def forward(self, x):
        mid=self.mid
        out1,mid = self.SRDB1(x,mid)
        out2,mid = self.SRDB2(out1,mid)
        out3,_ = self.SRDB3(out2,mid)

        return out3
##########################################################################


#####################################################The SDPED model###########################
class SDPED(nn.Module):

    def __init__(self,num_in_ch=3, num_out_ch=1,trade_SDB=1, num_feat=64, num_grow_ch=32,num_mid=21,num_block=5):
        super(SDPED, self).__init__()

        self.mid=num_mid

        self.conv_first_1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_first_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.fuse_first=nn.Conv2d(in_channels=num_feat,out_channels=self.mid,kernel_size=1)

        self.SRSRDB_list = nn.ModuleList([])
        self.fuse_list = nn.ModuleList([])
        for i in range (num_block):
            self.SRSRDB_list.append(CSDB(num_feat=num_feat,num_grow_ch=num_grow_ch,trade_SDB=trade_SDB))
            self.fuse_list.append(nn.Conv2d(in_channels=num_feat,out_channels=self.mid,kernel_size=1))

        self.conv_body_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_body_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.body_fuse = nn.Conv2d(in_channels=num_feat, out_channels=self.mid, kernel_size=1)

        self.fuse1=nn.Conv2d(in_channels=self.mid*(num_block+2),out_channels=512,kernel_size=3,padding=1)
        self.fuse2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1)
        self.fuse3 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.fuse4 = nn.Conv2d(in_channels=512, out_channels=num_out_ch, kernel_size=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.relu = nn.ReLU()

        self.apply(weight_init)

    def forward(self, x):

        feat = self.lrelu(self.conv_first_1(x))
        feat = self.lrelu(self.conv_first_2(feat))

        mid = [self.fuse_first(feat)]

        for i in range(len(self.SRSRDB_list)):
            feat = self.SRSRDB_list[i](feat)
            mid.append(self.fuse_list[i](feat))

        feat = self.lrelu(self.conv_body_1(feat))
        feat = self.relu(self.conv_body_2(feat))
        mid.append(self.body_fuse(feat))

        out=torch.cat(mid,dim=1)
        out=self.relu(self.fuse1(out))
        out = self.lrelu(self.fuse2(out))
        out = self.lrelu(self.fuse3(out))
        out = self.fuse4(out)
        out=F.sigmoid(out)

        result=out.clamp(0,1)

        return result
##############################################################


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda'
    print(device)

    Current_model = SDPED(trade_SDB=1,num_block=7)
    # checkpoint_path=''
    Current_model = nn.DataParallel(Current_model)
    # Current_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    get_parameter_number(Current_model)

    w=torch.rand(4,3,16,16).to(device)
    out=Current_model(w)
    print(out.shape)
