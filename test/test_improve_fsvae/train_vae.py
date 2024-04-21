import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))
import os

import torch
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
from snntorch import utils

from VaeInception1d import parse
from VaeInception1d import global_v as glv
from VaeInception1d import FSVAE, SNNInception1D
from VaeInception1d import SpikeEncoder,PoissonEncoder
from VaeInception1d import Datasets,DataTransformNrm,load_conf,register_dictloss



def train(network:FSVAE,trainloader,opti:torch.optim.Adam,epoch, spike_encoder:SpikeEncoder):
    n_steps = glv.network_config['n_steps']
    max_epoch = glv.network_config['epochs']
    
    mean_q_z = 0
    mean_p_z = 0
    mean_sampled_z = 0
    
    loss_table=[]
    loss_column=["epoch","batch_idx","loss","reconst","distance"]
    
    network = network.train()
    
    for batch_idx, data in enumerate(trainloader):  
        """
        x: [batch x channel x H x W] (※既に時間軸を捨てたものを渡す)
        """
        x,_=data #入力と正解ラベルに分ける

        spike_input=spike_encoder(x.to(DEVICE))
        spike_input=spike_input.permute(
            1,2,3,4,0
        ) #時間次元を一番最後に持っていく

        opti.zero_grad()

        #ここのx_reconは発火率（＝時間軸はなし）
        x_recon, q_z, p_z, sampled_z = network(spike_input.to(DEVICE), scheduled=glv.network_config['scheduled']) # sampled_z(B,C,1,1,T)
        
        if glv.network_config['loss_func'] == 'mmd':
            losses = network.loss_function_mmd(x.to(DEVICE), x_recon, q_z, p_z)
        elif glv.network_config['loss_func'] == 'kld':
            losses = network.loss_function_kld(x.to(DEVICE), x_recon, q_z, p_z)
            
        else:
            raise ValueError('unrecognized loss function')
        
        losses['loss'].backward()
        
        opti.step()
        network.weight_clipper()

        loss_table_idx=[epoch,batch_idx]+[loss.detach().cpu().item() for loss in losses.values()]
        loss_table=loss_table+[loss_table_idx]
        # print(loss_table)
        loss_pd=pd.DataFrame(loss_table,columns=loss_column)

        mean_q_z = (q_z.mean(0).detach().cpu() + batch_idx * mean_q_z) / (batch_idx+1) # (C,k,T)
        mean_p_z = (p_z.mean(0).detach().cpu() + batch_idx * mean_p_z) / (batch_idx+1) # (C,k,T)
        mean_sampled_z = (sampled_z.mean(0).detach().cpu() + batch_idx * mean_sampled_z) / (batch_idx+1) # (C,T)

        print(f'Epoch[{epoch}/{max_epoch}] Batch[{batch_idx}/{len(trainloader)}] Loss: {loss_pd["loss"].mean()}, RECONS: {loss_pd["reconst"].mean()}, DISTANCE: {loss_pd["distance"].mean()}')

    mean_q_z = mean_q_z.permute(1,0,2) # (k,C,T)
    mean_p_z = mean_p_z.permute(1,0,2) # (k,C,T)

    return loss_pd


def show_images(x, y,is_show=True, save_file_name=""):
    """
    結果を見る関数
    :param is_show: Falseにすると保存される
    """
    image_num=5
    fig, axs = plt.subplots(2, image_num, figsize=(2*image_num, 4))

    for i in range(image_num):
        index = random.randint(0, len(x) - 1)
        axs[0, i].imshow(x[index][0], cmap='gray')
        axs[0, i].axis('off')

        axs[1, i].imshow(y[index][0], cmap='gray')
        axs[1, i].axis('off')

    if is_show:
        plt.show()
    elif not is_show:
        plt.savefig(save_file_name)
    
    plt.close()


def plot_loss(data: pd.DataFrame, save_file_name=""):
    data.reset_index(inplace=True,drop=True)
    fig, ax1 = plt.subplots()
    alpha=0.5

    # 1つ目の y 軸に loss と reconst をプロット
    color = 'tab:red'
    ax1.set_xlabel('Batch Index')
    ax1.set_ylabel('Loss/Reconst', color=color)
    ax1.plot(data.index, data['loss'], color='red', label='loss',alpha=alpha)
    ax1.tick_params(axis='y', labelcolor=color)

    if "reconst" in data.columns.tolist():
        ax1.plot(data.index, data['reconst'], color='orange', label='reconst',alpha=alpha)

    # 2つ目の y 軸に distance をプロット
    if "distance" in data.columns.tolist():
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Distance', color=color)
        ax2.plot(data.index, data['distance'], color='blue', label='distance',alpha=alpha)
        ax2.tick_params(axis='y', labelcolor=color)

    # タイトルと凡例の追加
    fig.suptitle('Losses and Distance Over Batches')
    fig.legend()

    # グリッドを追加
    ax1.grid(True)    
    plt.savefig(save_file_name)
    plt.close()



def main():
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

    #>> FSVAEのconf >>
    params = parse(config_dir)
    network_config = params['VAE']
    glv.init(network_config, [DEVICE])
    #>> FSVAEのconf >>

    data_dir=conf["data_dir"]
    input_data:np.ndarray=np.load(f"{data_dir}/input_3d.npy")
    input_data=input_data[:,:,np.newaxis,:,:] #channel方向に次元を伸ばす
    label_data:np.ndarray=np.load(f"{data_dir}/label.npy").astype(int)
    print("input_shape : " + f"{input_data.shape}")
    print("label_shape : " + f"{label_data.shape}")


    # >> データのリサイズと標準化 >>
    data_size=(32,32) #28, 64
    n,t,c,h,w=input_data.shape
    transform=DataTransformNrm()
    input_data_nrm,max,min=transform(input_data.reshape(n*t,c,h,w),data_size,max=0.02,min=0)
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
        
    test_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[train_size:]],
        y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)
        )
    
    train_vae_loader = DataLoader(train_vae_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=conf["Inception"]["batch_size"], shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>



    #>> SNN-VAEを学習 >>
    fsvae:torch.nn.Module=FSVAE(hidden_dims=conf["VAE"]["hidden_dims"])
    spike_encoder=PoissonEncoder(snn_time_step=network_config["n_steps"])
    optimizer=torch.optim.Adam(params=fsvae.parameters(),lr=network_config["lr"])
    optimizer.param_groups[0]["caputurable"]=True

    if conf["is_vae_train"]:
        epochs=network_config["epochs"]
        train_loss_trj=pd.DataFrame([])
        for epoch in range(epochs):
            fsvae.train()

            if network_config['scheduled']:
                fsvae.update_p(epoch, glv.network_config['epochs'])

            train_loss=train(
                fsvae,train_vae_loader,optimizer,epoch,
                spike_encoder
            )

            train_loss_trj=pd.concat([
                train_loss_trj,train_loss
            ])
            train_loss_trj.to_csv(
                result_dir/"result_loss.csv",index=False
            )
            

            if (epoch+1)%conf["save_interval"]==0:
                torch.save(fsvae.to(DEVICE).state_dict(), result_dir/f"param{epoch}.pth")
                plot_loss(train_loss_trj,result_dir/f"loss_curve.png")
                #>> 1エポックごとにテストデータで評価してみる >>
                for i,data in enumerate(test_loader,0):
                    fsvae.eval()
                    x,_=data
                    n,t,c,h,w=x.shape
                    x=x.view(n*t,c,h,w) #時間軸を一旦忘れる
                    spike_x=x.unsqueeze(-1).repeat((1,1,1,1,network_config["n_steps"])) #時間次元を拡張(pop encとか入れたらいらない)
                    x_recon,_,_,_=fsvae(spike_x.to(DEVICE),scheduled=network_config["scheduled"])
                    show_images(x.to("cpu").detach().numpy(),x_recon.to("cpu").detach().numpy()
                                ,is_show=False, save_file_name=result_dir/f"img_epoch{epoch}.png")
                    break
                #>> 1エポックごとにテストデータで評価してみる >>
    else:
        print("*****load vae model*****")
        fsvae.load_state_dict(torch.load(conf["vae_model_path"]))
    #>> SNN-VAEを学習 >>
            
    
            

if __name__=="__main__":
    main()

