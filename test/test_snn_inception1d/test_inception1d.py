import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))

import torch
import torchvision
import torchvision.transforms as transforms
torch.set_default_tensor_type(torch.cuda.FloatTensor)
DEVICE=torch.Tensor([0,0]).device
print(DEVICE)

import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd

from VaeInception1d import SNNInception1D


def main():

    snn_threshold=1.0
    inc=SNNInception1D(
        in_channels=3,num_classes=10,
        snn_threshold=snn_threshold
    )

    T=16
    test_data=np.ones(T*5*3*8).reshape(T,5,3,8)
    print(test_data)
    print(test_data.shape)

    for t in range(T):
        print(t)
        out=inc(torch.Tensor(test_data[t]))
        print(out)
        print(out.shape)

if __name__=="__main__":
    main()