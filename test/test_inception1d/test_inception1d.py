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

from VaeInception1d import Inception1D


def main():

    inc=Inception1D(
        in_channels=3,num_classes=10
    )

    test_data=np.arange(5*3*8).reshape(5,3,8)
    print(test_data)
    print(test_data.shape)

    out=inc(torch.Tensor(test_data))
    print(out)
    print(out.shape)

if __name__=="__main__":
    main()