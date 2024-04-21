import torch
from torch import nn


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.branch3x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch5x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(nn.functional.avg_pool1d(x, kernel_size=3, stride=1, padding=1))
        
        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, dim=1)


class Inception1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Inception1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=1)
        self.inception_block1 = InceptionBlock(32, 32)
        self.inception_block2 = InceptionBlock(128, 64)
        self.fc = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.fc(x)
        return x