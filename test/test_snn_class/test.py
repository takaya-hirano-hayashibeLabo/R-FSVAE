"""
これを３次元にすれば行けそう
"""

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
PARENT=str(Path(__file__).parent)

import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

import torch
from VaeInception1d import Datasets,DataTransformNrm,load_conf,register_dictloss


# dataloader arguments
batch_size = 128
data_path=f"{PARENT}/data/mnist"

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



#>> poisson encoder >>
class SpikeEncoder():
    def __call__(self,x:torch.Tensor)->torch.Tensor:
        """
        アナログ入力をSNNの時間方向に引き伸ばす関数
        :param x [batch x channel x h x w]
        :return out [snn_time_steps x batch x channel x h x w]
        """
        return NotImplementedError()  

class PoissonEncoder(SpikeEncoder):

    def __init__(self,snn_time_step):
        super(PoissonEncoder,self).__init__()
        self.snn_time_step=snn_time_step

    def __call__(self,x:torch.Tensor)->torch.Tensor:
        """
        アナログ入力をSNNの時間方向に引き伸ばす関数
        :param x [batch x channel x h x w]
        :return out [snn_time_steps x batch x channel x h x w]
        """
        out=[torch.where(
            torch.rand(size=x.shape).to(device)<=x,
            1,
            0
        ) for _ in range(self.snn_time_step)]
        out=torch.stack(out).type(torch.float)

        return out
#>> poisson encoder >>


import argparse
import os
parser=argparse.ArgumentParser()
# parser.add_argument("--data_dir", required=True)
parser.add_argument("--target_dir",type=str,required=True)
# parser.add_argument("--is_vae_train",default=True) #VAEを学習するか？
# parser.add_argument("--vae_model_path",default=None) #is_vae_train=Trueのときは, モデルのファイルパスを設定
args=parser.parse_args()


#>> 個人用のconf >>
config_dir=args.target_dir+f"/conf.yml"
result_dir=Path(args.target_dir+f"/result")
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
conf=load_conf(config_dir)
print(conf)
#>> 個人用のconf >>


data_dir=conf["data_dir"]
input_data:np.ndarray=np.load(f"{data_dir}/input_3d.npy")
input_data=input_data[:,:,np.newaxis,:,:] #channel方向に次元を伸ばす
label_data:np.ndarray=np.load(f"{data_dir}/label.npy").astype(int)
print("input_shape : " + f"{input_data.shape}")
print("label_shape : " + f"{label_data.shape}")


# >> データのリサイズと標準化 >>
data_size=(conf["VAE"]["input_size"],conf["VAE"]["input_size"]) #28, 64
n,t,c,h,w=input_data.shape
transform=DataTransformNrm()
input_data_nrm,max,min=transform(input_data.reshape(n*t,c,h,w),data_size,max=0.01,min=0)
input_data_nrm=input_data_nrm.view(n,t,c,data_size[0],data_size[1])
print(np.max(input_data),np.min(input_data))
print(torch.max(input_data_nrm),torch.min(input_data_nrm))
# exit(1)
# >> データのリサイズと標準化 >>


#>> データのシャッフルと分割 >>
train_size_rate=conf["train_size_rate"]
batch_size=conf["batch_size"]
train_size=round(train_size_rate*input_data.shape[0]) #学習データのサイズ
print(f"train_size:{train_size}, test_size:{input_data.shape[0]-train_size}")
shuffle_idx=torch.randperm(input_data.shape[0])
# print(shuffle_idx)

#vae用のデータは時間軸を展開しておく
vae_y=torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32)\
        .view(-1,1).repeat(1,t).flatten() #時間軸に引き伸ばした正解ラベル
train_vae_dataset=Datasets(
    x=input_data_nrm[shuffle_idx[:train_size]].view(train_size*t,c,h,w),
    y=vae_y
    )
print(torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32))
print(vae_y)


vae_y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)\
        .view(-1,1).repeat(1,t).flatten() #時間軸に引き伸ばした正解ラベル
print(input_data_nrm[shuffle_idx[train_size:]].shape,vae_y.shape)
test_vae_dataset=Datasets(
    x=input_data_nrm[shuffle_idx[train_size:]].view(vae_y.shape[0],c,h,w),
    y=vae_y
    )

train_loader = DataLoader(train_vae_dataset, batch_size=conf["Inception"]["batch_size"], shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
test_loader = DataLoader(test_vae_dataset, batch_size=conf["Inception"]["batch_size"], shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
#>> データのシャッフルと分割 >>


# #Define a transform
# transform = transforms.Compose([
#             transforms.Resize((32,32)),
#             transforms.Grayscale(),
#             transforms.ToTensor(),
#             transforms.Normalize((0,), (1,))])

# mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
# mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

# # Create DataLoaders
# train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
# test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))



# neuron and simulation parameters
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5
num_steps = 8

    
#  Initialize Network
threshold=1.0

#>> プーリングのカーネルサイズをいい感じに設定 >>
if conf["VAE"]["input_size"]==128:
   kernel_pool=8
elif conf["VAE"]["input_size"]==32:
   kernel_pool=4
#>> プーリングのカーネルサイズをいい感じに設定 >>

net = nn.Sequential(nn.Conv2d(1, 16, 5,stride=1,padding=2),
                    nn.AvgPool2d(kernel_pool),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
                    nn.Conv2d(16,64, 5,padding=2),
                    nn.AvgPool2d(kernel_pool),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
                    nn.Flatten(),
                    nn.Linear(64*2*2, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=threshold)
                    ).to(device)

# net = nn.Sequential(nn.Conv2d(1, 16, 5,stride=1,padding=2),
#                     nn.AvgPool2d(4),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
#                     nn.Conv2d(16,32, 5,padding=2),
#                     nn.AvgPool2d(4),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
#                     nn.Conv2d(32, 64, 5,stride=1,padding=2),
#                     nn.AvgPool2d(4),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
#                     nn.Flatten(),
#                     nn.Linear(64*2*2, 10),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=threshold)
#                     ).to(device)

# net = nn.Sequential(nn.Conv2d(1, 256, 5,stride=1,padding=2),
#                     nn.AvgPool2d(16),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
#                     nn.Flatten(),
#                     nn.Linear(256*2*2, 10),
#                     snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=threshold)
#                     ).to(device)
    
encoder=PoissonEncoder(num_steps)
def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  spk_in=encoder(data.to(device)) #poisson encoder

  for step in range(num_steps):
      spk_out, mem_out = net(spk_in[step])
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)
  
  return torch.stack(spk_rec), torch.stack(mem_rec)

# already imported snntorch.functional as SF 
loss_fn = SF.ce_rate_loss()


def batch_accuracy(train_loader, net, num_steps):
  with torch.no_grad():
    total = 0
    acc = 0
    net.eval()
    
    train_loader = iter(train_loader)
    for data, targets in train_loader:
      data = data.to(device)
      targets = targets.to(device)
      spk_rec, _ = forward_pass(net, num_steps, data)

      acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
      total += spk_rec.size(1)

  return acc/total


optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999))
num_epochs = 30
loss_hist = []
test_acc_hist = []
counter = 0
skip_train=False

# Outer training loop
for epoch in range(num_epochs):

    # Training loop
    for data, targets in iter(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        if skip_train:
            break

        # forward pass
        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)
        # print(torch.sum(spk_rec,dim=0))
        # print("out_sp shape : ",spk_rec.shape, type(spk_rec),type(spk_rec[0,0,0].item()))
        # print("targets shape : ",targets.shape, type(targets),type(targets[0].item()))

        # initialize the loss & sum over time
        # spk_rec:float, targets:int
        loss_val = loss_fn(spk_rec.to(float), targets.to(int))

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        # for param in net.parameters():
        #     print(param.grad)
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        if counter % 50 == 0:
          with torch.no_grad():
              net.eval()
            #   print(torch.sum(spk_rec,dim=0))
            #   print(targets)
              # Test set forward pass
              test_acc = batch_accuracy(test_loader, net, num_steps)
              print(f"epoch {epoch} / Iteration {counter}, Test Acc: {test_acc * 100:.2f}%")
              test_acc_hist.append(test_acc.item())

        counter += 1

import pandas as pd
test_acc=pd.DataFrame(
    test_acc_hist,columns=["test_acc"]
)
test_acc.to_csv(result_dir/"test_acc.csv")
        
# # Plot Loss
fig = plt.figure(facecolor="w")
plt.plot(test_acc_hist)
plt.title("Test Set Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
fig.savefig(result_dir/"test_acc.png")
# plt.show()

# プロットを作成
import random
# プロットを作成
fig,axs=plt.subplots(3,3)

for axi in range(3):
   for axj in range(3):
        idx=random.randint(a=0,b=data.shape[0]-1)
        data_ax=data[idx,0].to("cpu").numpy()
    
        axs[axi,axj].imshow(np.fliplr(data_ax), cmap='gray', interpolation='nearest')  # 画像を表示
        axs[axi,axj].set_title(f"{targets[idx]}")

        # cbar =axs[axi,axj].colorbar(fraction=0.046, pad=0.04)  # fractionとpadでカラーバーのサイズを調整
        # cbar.set_label('Label')  # カラーバーのラベルを設定

        # 画像のピクセル位置にテキスト形式で描画
        # for i in range(data_ax.shape[0]):
        #     for j in range(data_ax.shape[1]):
        #         axs[axi,axj].text(j, i, '{:.2f}'.format(data_ax[i, j]), ha='center', va='center', color='red',fontsize=4)

# plt.imshow(np.fliplr(data), cmap='gray', interpolation='nearest')  # 画像を表示
# plt.title(f"{targets[idx]}")

# # カラーバーを表示
# cbar = plt.colorbar(fraction=0.046, pad=0.04)  # fractionとpadでカラーバーのサイズを調整
# cbar.set_label('Label')  # カラーバーのラベルを設定

# # 画像のピクセル位置にテキスト形式で描画
# for i in range(data.shape[0]):
#     for j in range(data.shape[1]):
#         plt.text(j, i, '{:.2f}'.format(data[i, j]), ha='center', va='center', color='red',fontsize=4)

# plt.show()  # プロットを表示
plt.savefig(result_dir/"input_fig.png")