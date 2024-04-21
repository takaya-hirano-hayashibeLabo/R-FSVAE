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
from snntorch import utils
import snntorch as snn
from snntorch import surrogate

from VaeInception1d import parse
from VaeInception1d import global_v as glv
from VaeInception1d import FSVAE, SNNInception1D
from VaeInception1d import SpikeEncoder,PoissonEncoder
from VaeInception1d import Datasets,DataTransformNrm,load_conf,register_dictloss


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
    
    #inception用のデータは時間軸を残しておく
    train_inception_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[:train_size]],
        y=torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32)
        )    
    
    test_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[train_size:]],
        y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)
        )
    
    train_inception_loader = DataLoader(train_inception_dataset, batch_size=conf["Inception"]["batch_size"], shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=conf["Inception"]["batch_size"], shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>



    #>> SNN-VAEを学習 >>
    fsvae:torch.nn.Module=FSVAE(hidden_dims=conf["VAE"]["hidden_dims"]).to(DEVICE)
    spike_encoder=PoissonEncoder(snn_time_step=network_config["n_steps"])

    print("*****load vae model*****")
    fsvae.load_state_dict(torch.load(conf["vae_model_path"]))
    #>> SNN-VAEを学習 >>
            
                
    spike_grad = surrogate.fast_sigmoid(slope=25)
    # inception=nn.Sequential(
    #         nn.Conv1d(256, 256, kernel_size=5, padding=2),
    #         nn.AvgPool1d(4),
    #         snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=1.0),
    #         nn.Conv1d(256, 256, kernel_size=5, padding=2),
    #         nn.AvgPool1d(4),
    #         snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=1.0),
    #         nn.Flatten(),
    #         nn.Linear(256,conf["Inception"]["num_class"]),
    #         snn.Leaky(beta=0.5, spike_grad=spike_grad, init_hidden=True,threshold=1.0,output=True),
    # ).to(DEVICE)
    

    # convのチャンネルを1024にしても意味がない
    threshold=0.1
    beta=0.5
    inc_channels=conf["Inception"]["latent_dim"]
    inception = nn.Sequential(nn.Conv1d(inc_channels, inc_channels, 3,stride=1,padding=1),
                        nn.AvgPool1d(4),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
                        nn.Conv1d(inc_channels,inc_channels, 3,padding=1,stride=1),
                        nn.AvgPool1d(4),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
                        nn.Flatten(),
                        nn.Linear(inc_channels, 10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=threshold)
                        ).to(DEVICE)
    
    optimizer=torch.optim.Adam(
        params=inception.parameters(),lr=conf["Inception"]["lr"],
        betas=(0.9,0.999)
        )
    # optimizer.param_groups[0]["caputurable"]=True
    criterion=SF.ce_rate_loss()
    # criterion=torch.nn.CrossEntropyLoss()
    epochs=conf["Inception"]["epochs"]
    loss_db=pd.DataFrame([])
    
    
    fsvae.eval() #vaeは学習しない
    for epoch in tqdm(range(epochs)):
        train_loss=0
        for i,data in enumerate(train_inception_loader):
            
            x,label=data

            n,t,c,h,w=x.shape
            x=x.view(n*t,c,h,w) #時間軸を一旦忘れる

            # #プロットを作成
            # fig,axs=plt.subplots(4,4)
            # label_idx=0
            # x_t=0
            # for axi in range(4):
            #     for axj in range(4):
            #         data_ax=x[x_t,0].to("cpu").numpy()  
            #         axs[axi,axj].imshow(np.fliplr(data_ax), cmap='gray', interpolation='nearest')  # 画像を表示
            #         axs[axi,axj].set_title(f"{label[label_idx]}")
            #         x_t+=1
            # fig.savefig(result_dir/"label_check_fig.png")
            # exit(1)


            with torch.no_grad():
                spike_input=spike_encoder(x.to(DEVICE))

                # fig,axs=plt.subplots(4,4)
                # label_idx=0
                # x_t=0
                # for axi in range(4):
                #     for axj in range(4):
                #         data_ax=torch.sum(spike_input[:,x_t,0],dim=0).to("cpu").numpy()
                #         axs[axi,axj].imshow(np.fliplr(data_ax), cmap='gray', interpolation='nearest')  # 画像を表示
                #         axs[axi,axj].set_title(f"{label[label_idx]}")
                #         x_t+=1
                # fig.savefig(result_dir/"label_check_fig.png")
                # exit(1)
                
                spike_input=spike_input.permute(
                    1,2,3,4,0
                ) #時間次元を一番最後に持っていく
                z,_,_=fsvae.encode(spike_input.to(DEVICE)) #vaeにより特徴に変換. z[n*t x latent_dim x snn_step]
                recons=fsvae.decode(z) #[n*t x c x h x w]        

            inception.train()
            z:torch.Tensor=z.view(n,t,conf["VAE"]["latent_dim"],conf["VAE"]["n_steps"]) #時間軸の復元
            z=z.permute(3,0,2,1) #[snn_steps x batch x latent x time_seq] snn軸を一番外 & 特徴軸をチャンネルに持っていく
                        
            fig,axs=plt.subplots(2,4)
            label_idx=0
            x_t=0
            recons=recons.view(n,t,c,h,w)
            for axi in range(2):
                for axj in range(4):
                    data_ax=recons[label_idx,x_t,0].to("cpu").numpy()
                    axs[axi,axj].imshow(np.fliplr(data_ax), cmap='gray', interpolation='nearest')  # 画像を表示
                    axs[axi,axj].set_title(f"{label[label_idx]}",fontsize=6)
                    x_t+=1
            fig.savefig(result_dir/f"label{int(label[label_idx])}_z.png")
            exit(1)
            
            #>> snntorchを使うときはforwardの前にSNNtorchを使ってる層をリセットしないとダメ >>
            utils.reset(inception) 
            #>> snntorchを使うときはforwardの前にSNNtorchを使ってる層をリセットしないとダメ >>

            out_spikes=[] #[snn_steps x batch x class]
            for t_snn in range(conf["VAE"]["n_steps"]):
                # print(z[t_snn]) #ちゃんと出てる
                out,_=inception(z[t_snn]) #スパイク出力
                out_spikes.append(out)
            out_spikes=torch.stack(out_spikes).to(float)
            label=label.to(int)
            print("out_sp shape : ",out_spikes.shape, type(out_spikes),type(out_spikes[0,0,0].item()))
            print("label shape : ",label.shape, type(label),type(label[0].item()))
            loss=criterion(out_spikes,label.to(DEVICE))
            print(torch.sum(out_spikes,dim=0))
            print(label)

            optimizer.zero_grad()
            loss.backward()
            # for param in inception.parameters():
            #     print(param.grad)
            optimizer.step()
            train_loss+=loss.item()

            loss_db=register_dictloss(
                loss_db,epoch,i,
                {"loss":loss.item()}
            )
            
            # model_weights = inception.state_dict()
            # print("\n"+"="*50+"model weight"+"="*50)
            # for param_name, param_value in model_weights.items():
            #     print(f"Parameter: {param_name}, Shape: {param_value.shape}, Values: {param_value}")


        if (epoch)%conf["save_interval"]==0:
            print(f'[{epoch + 1}] loss: {loss.item():.3f}')

            loss_db.to_csv(
                result_dir/"loss_inception.csv",index=False
            )
            plot_loss(
                loss_db,save_file_name=result_dir/"loss_inception_trj.png"
            )

            fsvae.eval()  # モデルを評価モードに設定
            with torch.no_grad():
                inception.eval()
                for i, data in enumerate(test_loader, 0):
                    x,label=data
                    n,t,c,h,w=x.shape
                    x=x.view(n*t,c,h,w) #時間軸を一旦忘れる
                    z,_,_=fsvae.encode(spike_input.to(DEVICE)) #vaeにより特徴に変換. z[n*t x latent_dim x snn_step]

                    z:torch.Tensor=z.view(n,t,conf["VAE"]["latent_dim"],conf["VAE"]["n_steps"]) #時間軸の復元
                    z=z.permute(3,0,2,1) #[snn_steps x batch x latent x time_seq] 時間軸を一番外 & 特徴軸をチャンネルに持っていく
                    out_spikes=[] #[snn_steps x batch x class]
                    for t_snn in range(conf["VAE"]["n_steps"]):
                        out,_=inception(z[t_snn]) #スパイク出力
                        out_spikes+=[out]
                    out_spikes=torch.stack(out_spikes)
                    y=torch.mean(out_spikes,dim=0) #snn_step方向で平均し、発火率へ変換
                    break

            x=x.view(n,t,c,h,w) #時間軸復元
            classes=[i for i in range(10)]

            #>> 正解ラベルと予測確率を描画 >>
            # プロットの初期化
            n=5 #描画する要素数
            fig, ax = plt.subplots(2, n, figsize=(12,6))

            # 各要素ごとに正解ラベルと予測確率を描画
            for idx in range((n)):
                # 入力画像の描画
                ax[0][idx].imshow(x[idx][-1][0].to("cpu").detach().numpy())  # 入力画像を表示
                ax[0][idx].set_title(f'True Label: {classes[int(label[idx].item())]}')
                ax[0][idx].axis('off')

                # 予測確率の描画
                probs = y[idx].softmax(dim=0)  # ソフトマックス関数を適用して確率に変換
                ax[1][idx].barh(classes, probs.cpu().detach().numpy())  # 各クラスの確率を水平棒グラフで表示
                ax[1][idx].set_xlabel('Probability')
                ax[1][idx].set_xlim(0, 1)

            fig.savefig(result_dir/f"estimate_class_epoch{epoch+1}.png")
            plt.close()
            #>> 正解ラベルと予測確率を描画 >>
            
            
            #>> 潜在スパイクを描画 >>
            n=5 #描画する要素数
            fig, ax = plt.subplots(n,1, figsize=(12,6))
            classes=[i for i in range(10)]

            # 各要素ごとに正解ラベルと予測確率を描画
            for idx in range((n)):
                ax[idx].imshow(torch.sum(z[:,idx,:,:],dim=0).permute(1,0).to("cpu").numpy(),cmap="hot")
                ax[idx].set_title(f'True Label: {classes[int(label[idx].item())]}')
            fig.savefig(result_dir/f"z_heat_epoch{epoch+1}.png")
            plt.close()
            #>> 潜在スパイクを描画 >>
            
    #>> Inceptionの学習 >>
            

if __name__=="__main__":
    main()

