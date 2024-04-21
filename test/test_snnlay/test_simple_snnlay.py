import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))
import os

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from torch.utils.data import DataLoader
DEVICE=torch.Tensor([0,0]).device
print(DEVICE)

import matplotlib.pyplot as plt
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

from snntorch import functional as SF
from VaeInception1d import parse
from VaeInception1d import global_v as glv
from VaeInception1d import FSVAE, SNNInception1D
from VaeInception1d import tdLinear,LIFSpike,tdBatchNorm
from VaeInception1d import SpikeEncoder,PoissonEncoder
from VaeInception1d import Datasets,DataTransformNrm,load_conf,register_dictloss





def main():
    import os
    parser=argparse.ArgumentParser()
    # parser.add_argument("--data_dir", required=True)
    parser.add_argument("--target_dir",type=str,required=True)
    # parser.add_argument("--is_vae_train",default=True) #VAEを学習するか？
    # parser.add_argument("--vae_model_path",default=None) #is_vae_train=Trueのときは, モデルのファイルパスを設定
    args=parser.parse_args()
    
    #>> 個人用のconf >>
    result_dir=Path(args.target_dir+f"/result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    #>> 個人用のconf >>
    
    
    n,l,t=10,4,3
    snn_steps=8
    z=torch.stack(
        [torch.where(
            torch.rand(size=(n,l,t))>0.7,
            1,0
        ) for _ in range(snn_steps)]
    )
    z=z.permute(1,2,3,0)
    z=z.reshape(n,l*t,snn_steps).to(torch.float)
    
    
    label=torch.randint(low=0,high=9,size=(n,))
    
    
    net=tdLinear(
            l*t,10,
            spike=LIFSpike(),
            # bn=tdBatchNorm(10)
            )
    
    
    optimizer=torch.optim.Adam(
        params=net.parameters(),lr=0.001,
        betas=(0.9,0.999)
        )
    # optimizer.param_groups[0]["caputurable"]=True
    criterion=SF.ce_rate_loss()
    
    
    for i in range(1000):
        out_spike=net(z)   
        loss=criterion(out_spike,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i%100==0:
            print(f"{'='*25}{i}{'='*25}")
            print(torch.sum(out_spike,dim=-1))
            print(label)


    
    #>> 潜在スパイクを描画 >>
    n=5 #描画する要素数
    fig, ax = plt.subplots(n,1, figsize=(12,6))
    classes=[i for i in range(10)]

    # 各要素ごとに正解ラベルと予測確率を描画
    for idx in range((n)):
        ax[idx].imshow(torch.sum(z[:,idx,:,:],dim=0).permute(1,0).to("cpu").numpy(),cmap="plasma")
        ax[idx].set_title(f'True Label: ')
    fig.savefig(result_dir/f"z_heat.png")
    plt.close()
    #>> 潜在スパイクを描画 >>

if __name__=="__main__":
    main()