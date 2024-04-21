import torch
from torch import nn

import snntorch as snn
from snntorch import surrogate
from snntorch import utils

class SNNInceptionBlock(nn.Module):
    """
    畳み込み軸の要素数を半分にする
    """
    def __init__(self, in_channels, out_channels,snn_threshold=1.0):
        super(SNNInceptionBlock, self).__init__()

        spike_grad=surrogate.fast_sigmoid()
        self.branch1x1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(2),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold),
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.AvgPool1d(2),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold),
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.AvgPool1d(2),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold),
        )
        self.branch_pool = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool1d(2),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold)
        )
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1))
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]

        return torch.cat(outputs, dim=1)


class SNNInception1D(nn.Module):
    def __init__(self, in_channels, num_classes, snn_threshold=1.0):
        super(SNNInception1D, self).__init__()

        spike_grad=surrogate.fast_sigmoid()
        self.conv1 =nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=1),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold)
        )
        self.inception_block1 = SNNInceptionBlock(32, 32 ,snn_threshold)
        self.inception_block2 = SNNInceptionBlock(128, 64,snn_threshold)
        
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold),
            nn.Linear(64, num_classes),
            snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=snn_threshold,output=True)

        )


    def forward(self, x):
        # print("in: ",x.shape,torch.sum(x).item())
        # print(torch.sum(torch.sum(x,dim=-1),dim=-1))
        x = self.conv1(x)
        # print("conv1: ",x.shape,torch.sum(x).item())
        x = self.inception_block1(x)
        # print("inc1: ",x.shape,torch.sum(x).item())
        x = self.inception_block2(x)
        # print("inc2: ",x.shape,torch.sum(x).item())
        x = torch.mean(x, dim=2)  # Global average pooling.
        x=torch.flatten(x,start_dim=1)
        # print("glv pool: ",x.shape,torch.sum(x).item())
        sp,mem = self.fc(x)
        # print("fc: ",sp.shape,torch.sum(sp).item())
        # print(sp)

        return sp