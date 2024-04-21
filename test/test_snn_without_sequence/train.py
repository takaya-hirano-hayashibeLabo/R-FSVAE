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
from VaeInception1d import Datasets,DataTransformNrm,load_conf,register_dictloss,register_dictdata

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
    
    #時間軸を展開しておく
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
    
    train_loader = DataLoader(train_vae_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>



    fsvae:torch.nn.Module=FSVAE(hidden_dims=conf["VAE"]["hidden_dims"]).to(DEVICE)
    spike_encoder=PoissonEncoder(snn_time_step=network_config["n_steps"])

    print("*****load vae model*****")
    fsvae.load_state_dict(torch.load(conf["vae_model_path"]))
            
                
    spike_grad = surrogate.fast_sigmoid(slope=25)
    

    #>> 特徴量からシンプルな層で分類してみる >>
    threshold=1.0
    beta=0.5
    inc_channels=conf["Inception"]["latent_dim"]
    test_model = nn.Sequential(nn.Linear(inc_channels,256),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
                        nn.Linear(256,128),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
                        nn.Linear(128,10),
                        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True,threshold=threshold)
                        ).to(DEVICE)
    
    optimizer=torch.optim.Adam(
        params=test_model.parameters(),lr=conf["Inception"]["lr"],
        betas=(0.9,0.999)
        )
    # optimizer.param_groups[0]["caputurable"]=True
    criterion=SF.ce_rate_loss()
    # criterion=torch.nn.CrossEntropyLoss()
    epochs=conf["Inception"]["epochs"]
    loss_db=pd.DataFrame([])
    acc_db=pd.DataFrame([])
    
    
    fsvae.eval() #vaeは学習しない
    for epoch in (range(epochs)):
        train_loss=0

        print("-"*40+f"epoch[{epoch}] train model"+"-"*40)
        for i,data in tqdm(enumerate(train_loader)):
            
            x,label=data

            with torch.no_grad():
                spike_input=spike_encoder(x.to(DEVICE))                
                spike_input=spike_input.permute(
                    1,2,3,4,0
                ) #時間次元を一番最後に持っていく
                z,_,_=fsvae.encode(spike_input.to(DEVICE)) #vaeにより特徴に変換. z[n*t x latent_dim x snn_step]

            test_model.train()
            z=z.permute(-1,0,1) #[snn_steps x batch x latent x time_seq] snn軸を一番外に持っていく
                        
            
            #>> snntorchを使うときはforwardの前にSNNtorchを使ってる層をリセットしないとダメ >>
            utils.reset(test_model) 
            #>> snntorchを使うときはforwardの前にSNNtorchを使ってる層をリセットしないとダメ >>

            out_spikes=[] #[snn_steps x batch x class]
            for t_snn in range(conf["VAE"]["n_steps"]):
                out,_=test_model(z[t_snn]) #スパイク出力
                out_spikes.append(out)
            out_spikes=torch.stack(out_spikes).to(float)
            label=label.to(int)
            loss=criterion(out_spikes,label.to(DEVICE))
            # print("out_sp shape : ",out_spikes.shape, type(out_spikes),type(out_spikes[0,0,0].item()))
            # print("label shape : ",label.shape, type(label),type(label[0].item()))
            # print(torch.sum(out_spikes,dim=0))
            # print(label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()

            loss_db=register_dictloss(
                loss_db,epoch,i,
                {"loss":loss.item()}
            )

        print("-"*40+f"epoch[{epoch}] train done"+"-"*40)


        if (epoch)%conf["save_interval"]==0:
            print(f'[{epoch + 1}] loss: {loss.item():.3f}')

            loss_db.to_csv(
                result_dir/"loss_inception.csv",index=False
            )
            plot_loss(
                loss_db,save_file_name=result_dir/"loss_inception_trj.png"
            )

            fsvae.eval()  # モデルを評価モードに設定
            print("-"*40+f"epoch[{epoch}] eval model"+"-"*40)
            with torch.no_grad():
                test_model.eval()
                acc=0
                test_data_num=0
                for i, data in tqdm(enumerate(test_loader)):
                    x,label=data
                    spike_input=spike_encoder(x.to(DEVICE))                
                    spike_input=spike_input.permute(
                        1,2,3,4,0
                    ) #snn次元を一番最後に持っていく
                    z,_,_=fsvae.encode(spike_input.to(DEVICE)) #vaeにより特徴に変換. z[n*t x latent_dim x snn_step]

                    z=z.permute(-1,0,1) #[snn_steps x batch x latent] 時間軸を一番外 & 特徴軸をチャンネルに持っていく
                    out_spikes=[] #[snn_steps x batch x class]

                    #>> snntorchを使うときはforwardの前にSNNtorchを使ってる層をリセットしないとダメ >>
                    utils.reset(test_model) 
                    #>> snntorchを使うときはforwardの前にSNNtorchを使ってる層をリセットしないとダメ >>
                    
                    for t_snn in range(conf["VAE"]["n_steps"]):
                        out,_=test_model(z[t_snn]) #スパイク出力
                        out_spikes+=[out]
                    out_spikes=torch.stack(out_spikes)
                    y=torch.sum(out_spikes,dim=0) #snn_step方向で平均し、発火数へ変換

                    predict_label=torch.argmax(
                        y,dim=1
                    ).flatten()
                    correct_num=torch.sum(
                        label.flatten()==predict_label.flatten()
                    )
                    acc+=correct_num.item()
                    test_data_num+=x.shape[0]

                    print("iter : ",i," acc : ",acc/test_data_num*100,"%")

                recons=fsvae.decode(z.permute(1,2,0)) #描画用に一回だけ再構成

            print(f"-"*40+f"epoch[{epoch}] eval done"+"-"*40)
            acc=acc/test_data_num
            acc_db=register_dictdata(
                acc_db,epoch=epoch,iter=-1,
                data_dict={"acc":acc}
            )
            acc_db.to_csv(result_dir/"acc.csv",index=False)

            plt.plot(acc_db["acc"],color="orange")
            plt.title("test accuracy")
            plt.savefig(result_dir/"acc.png")
            plt.close()


            n=5 #描画する要素数
            fig, ax = plt.subplots(3, n, figsize=(12,6))
            for idx in range(n):
                # 入力画像の描画
                ax[0][idx].imshow(np.fliplr(x[idx][0].to("cpu").detach().numpy()))  # 入力画像を表示
                ax[0][idx].set_title(f'True img: label {int(label[idx].item())}')
                ax[0][idx].axis('off')

                ax[1][idx].imshow(np.fliplr(recons[idx][0].to("cpu").detach().numpy()))  # 入力画像を表示
                ax[1][idx].set_title(f'Reconst img')
                ax[1][idx].axis('off')

                classes=[i for i in range(10)]
                probs = y[idx].flatten().softmax(dim=0)  # ソフトマックス関数を適用して確率に変換
                # print(y[idx])
                # print(probs)
                bars=ax[2][idx].barh(classes, probs.cpu().detach().numpy())  # 各クラスの確率を水平棒グラフで表示
                ax[2][idx].set_xlabel('Probability')
                ax[2][idx].set_xlim(0, 1)
            
                # barの先端に発火数をテキストとして表示
                for bar, prob, spike_rate in zip(bars, probs,list(y[idx].to("cpu").detach().numpy().flatten())):
                    ax[2][idx].text(prob+0.15, bar.get_y() + bar.get_height()/2, f'{int(spike_rate)}', ha='right', va='center')

            fig.savefig(result_dir/f"estimate_class_epoch{epoch+1}.png")
            plt.close()
            #>> 正解ラベルと予測確率を描画 >>
                        
    #>> Inceptionの学習 >>
            

if __name__=="__main__":
    main()

