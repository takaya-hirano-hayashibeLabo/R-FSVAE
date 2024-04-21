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
from VaeInception1d import Datasets,DataTransformNrm,load_conf,register_dictdata,show_pred_result

from VaeInception1d import RecogFSVAE



def train(network:RecogFSVAE,trainloader,opti:torch.optim.Adam,epoch,alpha=0.0001):
    """
    :param alpha: 予測誤差にかける係数
    """
    max_epoch = glv.network_config['epochs']
    loss_pd=pd.DataFrame([])    
    network = network.train()
    
    for batch_idx, data in enumerate(trainloader):  
        """
        x: [batch x channel x H x W] (※既に時間軸を捨てたものを渡す)
        """
        x,label=data #入力と正解ラベルに分ける

        # ここのx_reconは発火率（＝時間軸はなし）
        x_recon, q_z, p_z, z, pred_spikes = network.forward(x) # sampled_z(B,C,1,1,T)
        loss_recon,loss_dist,loss_pred=network.loss_func(
            x,x_recon,q_z,p_z,pred_spikes,label
        )
        loss:torch.Tensor=loss_recon+loss_dist+alpha*loss_pred #トータル誤差

        loss.backward()
        opti.step()
        opti.zero_grad()
        network.fsvae.weight_clipper()

        loss_pd=register_dictdata(
            loss_pd,epoch=epoch,iter=batch_idx,
            data_dict={
                "loss":loss.item(),"reconst":loss_recon.item(),
                "distance":loss_dist.item(),"predict":loss_pred.item()
            }
        )
        
        print(f'Epoch[{epoch}/{max_epoch}] Batch[{batch_idx}/{len(trainloader)}]Loss: {loss_pd["loss"].mean()}, RECONS: {loss_pd["reconst"].mean()},DISTANCE: {loss_pd["distance"].mean()}, PREDICT:{loss_pd["predict"].mean()}')

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



def plot_loss(data:pd.DataFrame,save_file_name):
    # データのインデックスをx軸に使用
    data.reset_index(inplace=True,drop=True)
    x = data.index

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # 各列を異なる軸にプロット
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, column in enumerate(data.columns[2:]):
        row = idx // 2
        col = idx % 2
        ax = axes[row][col]
        ax.plot(x, data[column],color=colors[idx])
        ax.set_xlabel('Index')
        ax.set_ylabel(column)
        ax.set_title(column)

    plt.tight_layout()
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
    
    #データは時間軸を展開しておく
    vae_y=torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32)\
            .view(-1,1).repeat(1,t).flatten() #時間軸に引き伸ばした正解ラベル
    train_vae_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[:train_size]].view(train_size*t,c,h,w),
        y=vae_y
        )
        
    vae_y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)\
            .view(-1,1).repeat(1,t).flatten() #時間軸に引き伸ばした正解ラベル
    test_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[train_size:]].view(-1,c,h,w),
        y=vae_y
        )
    
    train_vae_loader = DataLoader(train_vae_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=conf["Inception"]["batch_size"], shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>



    #>> SNN-VAEを学習 >>
    rfsvae:torch.nn.Module=RecogFSVAE(
        spike_encoder=PoissonEncoder(network_config["n_steps"]),
        fsvae_hiddens=conf["VAE"]["hidden_dims"],
        recog_hiddens=conf["recog_hiddens"],
        device=DEVICE
        )
    optimizer=torch.optim.Adam(params=rfsvae.parameters(),lr=network_config["lr"])
    optimizer.param_groups[0]["caputurable"]=True

    if conf["is_vae_train"]:
        epochs=network_config["epochs"]
        train_loss_trj=pd.DataFrame([])
        acc_trj=pd.DataFrame([])
        for epoch in range(epochs):
            rfsvae.train()

            if network_config['scheduled']:
                rfsvae.fsvae.update_p(epoch, glv.network_config['epochs'])

            train_loss=train(
                rfsvae,train_vae_loader,optimizer,epoch,alpha=conf["alpha"]
            )

            train_loss_trj=pd.concat([
                train_loss_trj,train_loss
            ])
            train_loss_trj.to_csv(
                result_dir/"result_loss.csv",index=False
            )
            

            if (epoch+1)%conf["save_interval"]==0:

                torch.save(rfsvae.to(DEVICE).state_dict(), result_dir/f"param{epoch}.pth")
                plot_loss(train_loss_trj,result_dir/f"loss_curve.png")

                rfsvae.eval()
                acc=0
                test_data_num=0
                for i,data in enumerate(test_loader,0):          
                    x,label=data
                    x_recon, _,_,_, pred_spikes = rfsvae.forward(x) # sampled_z(B,C,1,1,T)

                    y=torch.sum(pred_spikes,dim=0) #snn_step方向で平均し、発火数へ変換
                    predict_label=torch.argmax(
                        y,dim=1
                    ).flatten()
                    correct_num=torch.sum(
                        label.flatten()==predict_label.flatten()
                    )
                    acc+=correct_num.item()
                    test_data_num+=x.shape[0]    

                acc_trj=register_dictdata(
                    acc_trj,epoch=epoch,iter=-1,
                    data_dict={"acc":acc/test_data_num}
                )
                acc_trj.to_csv(result_dir/"acc.csv",index=False)

                plt.plot(acc_trj["acc"],color="orange")
                plt.title("test accuracy")
                plt.savefig(result_dir/"acc.png")
                plt.close()

                show_pred_result(
                    x,x_recon,y,label,
                    file_name=result_dir/f"predict_result_epoch{epoch}.png",n=5
                )

    else:
        pass
    #>> SNN-VAEを学習 >>
            
    
            

if __name__=="__main__":
    main()

