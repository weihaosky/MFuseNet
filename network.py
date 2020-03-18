import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import numpy as np


def convbn2d(in_planes, out_planes, kernel_size, stride, padding, dilation=1):
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else padding, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn3d(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

def convTbn3d(in_planes, out_planes, kernel_size, stride, padding, output_padding):
    return nn.Sequential(nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride,  output_padding=output_padding, bias=False),
                         nn.BatchNorm3d(out_planes))

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out


class FuseNetwork(nn.Module):
    def __init__(self, DispRange, N_c):
        super(FuseNetwork, self).__init__()

        self.DispRange = DispRange
        self.N_c = N_c
        self.feature = nn.Sequential(
            convbn3d(1, 4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            convbn3d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            convbn3d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            convbn3d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        # ========= fuse ===========
        self.concat = convbn3d(4*self.N_c, 4, kernel_size=7, stride=1, padding=3)

        self.down1 = convbn3d(4, 4, kernel_size=3, stride=2, padding=1)
        self.dconv1 = convbn3d(4, 4, kernel_size=3, stride=1, padding=1)
        self.down2 = convbn3d(4, 4, kernel_size=3, stride=2, padding=1)
        self.dconv21 = convbn3d(4, 4, kernel_size=7, stride=1, padding=3)
        self.dconv22 = convbn3d(4, 4, kernel_size=3, stride=1, padding=1)

        self.up1 = convTbn3d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv1 = convbn3d(8, 4, kernel_size=3, stride=1, padding=1)
        self.up2 = convTbn3d(4, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.uconv2 = convbn3d(8, 4, kernel_size=7, stride=1, padding=3)

        self.distill1 = convbn3d(4, 4, kernel_size=5, stride=1, padding=2)
        self.classify = nn.Conv3d(4, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        costs = []
        for i in range(self.N_c):
            c1 = x[i].unsqueeze(1)
            c1 = self.feature(c1)
            costs.append(c1)

        # c = torch.min(costs)      # minimum fusion
        c = torch.cat(costs, 1)     # concatenation
        
        c = F.relu(self.concat(c))

        cdown1 = F.relu(self.down1(c))
        cdown1 = F.relu(self.dconv1(cdown1)) + cdown1

        cdown2 = F.relu(self.down2(cdown1))
        cdown2 = F.relu(self.dconv21(cdown2)) + cdown2
        cdown2 = F.relu(self.dconv22(cdown2)) + cdown2

        cup1 = torch.cat( (F.relu(self.up1(cdown2)), cdown1), 1)
        cup1 = F.relu(self.uconv1(cup1))

        cup2 = torch.cat( (F.relu(self.up2(cup1)), c), 1)
        cup2 = F.relu(self.uconv2(cup2))

        c = F.relu(self.distill1(cup2)) + cup2
        c = self.classify(c).squeeze(1)
        pred = F.softmax(c, dim=1)
        outdisp = disparityregression(self.DispRange)(pred)

        return outdisp


class FuseNetworkCAT(nn.Module):
    def __init__(self, DispRange, N_c):
        super(FuseNetwork, self).__init__()

        self.DispRange = DispRange
        self.N_c = N_c
        self.feature = nn.Sequential(
            convbn3d(1, 4, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            convbn3d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            convbn3d(4, 4, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            convbn3d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        # ========= fuse ============
        self.concat = convbn3d(4*self.N_c, 8, kernel_size=7, stride=1, padding=3)
        # self.distill1 = convbn3d(8, 8, kernel_size=3, stride=1, padding=1)
        self.distill2 = convbn3d(8, 8, kernel_size=5, stride=1, padding=2)
        self.distill3 = convbn3d(8, 8, kernel_size=5, stride=1, padding=2)
        self.classify = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        costs = []
        for i in range(self.N_c):
            c1 = x[i].unsqueeze(1)
            c1 = self.feature(c1)
            costs.append(c1)

        c = torch.cat(costs, 1)
        c = F.relu(self.concat(c))
        
        # c = F.relu(self.distill1(c)) + c
        c = F.relu(self.distill2(c)) + c
        c = F.relu(self.distill3(c)) + c

        c = self.classify(c).squeeze(1)
        pred = F.softmax(c, dim=1)
        outdisp = disparityregression(self.DispRange)(pred)

        return outdisp