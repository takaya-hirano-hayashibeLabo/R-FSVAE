import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent.parent
sys.path.append(str(ROOT))
sys.path.append(str(ROOT.parent))

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

torch.set_default_tensor_type(torch.cuda.FloatTensor)
DEVICE=torch.Tensor([0,0]).device
print(DEVICE)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

from VaeInception1d import VAE,register_loss,Inception1D,register_dictloss
from VaeInception1d import Datasets,DataTransformNrm,load_conf

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
    args=parser.parse_args()

    config_dir=args.target_dir+f"/conf.yml"
    result_dir=Path(args.target_dir+f"/result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    conf=load_conf(config_dir)
    print(conf)


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
    input_data_nrm,max,min=transform(input_data.reshape(n*t,c,h,w),data_size,max=0.04,min=0)
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
    
    #inception用のデータは時間軸を残しておく
    train_inception_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[:train_size]],
        y=torch.Tensor(label_data)[shuffle_idx[:train_size]].type(torch.float32)
        )    
    
    test_dataset=Datasets(
        x=input_data_nrm[shuffle_idx[train_size:]],
        y=torch.Tensor(label_data)[shuffle_idx[train_size:]].type(torch.float32)
        )
    
    train_vae_loader = DataLoader(train_vae_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    train_inception_loader = DataLoader(train_inception_dataset, batch_size=batch_size, shuffle=True, drop_last=True,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True  ,generator=torch.Generator(device=torch.Tensor([0,0]).device))
    #>> データのシャッフルと分割 >>


    #>> VAEの学習 >>
    vae=VAE(
        in_channel=conf["VAE"]["in_channel"],
        channels=conf["VAE"]["channels"],
        latent_dim=conf["VAE"]["latent_dim"],
        pool_sizes=conf["VAE"]["pool_sizes"]
    )
    optimizer=torch.optim.Adam(params=vae.parameters(),lr=conf["VAE"]["lr"])
    optimizer.param_groups[0]["caputurable"]=True
    epochs=conf["VAE"]["epochs"]
    loss_db=pd.DataFrame([])
    for epoch in tqdm(range(epochs)):
        vae.train()
        train_loss=0
        for i,data in enumerate(train_vae_loader):
            x,_=data
            y,z,kl_div,loss_rec=vae(x.to(DEVICE))
            loss:torch.Tensor=(loss_rec+kl_div)
            loss.backward()
            train_loss+=loss.item()

            optimizer.step()
            optimizer.zero_grad()

            loss_db=register_loss(
                loss_db,epoch,i,
                loss.item(),loss_rec.item(),kl_div.item()
            )

        if (epoch+1)%conf["save_interval"]==0:
            print(f'[{epoch + 1}] loss: {loss:.3f}')

            loss_db.to_csv(
                result_dir/"loss_vae.csv",index=False
            )
            plot_loss(
                loss_db,save_file_name=result_dir/"loss_vae_trj.png"
            )

            vae.eval()  # モデルを評価モードに設定
            with torch.no_grad():
                for i, data in enumerate(test_loader, 0):
                    x, _ = data
                    n,t,c,h,w=x.shape
                    x=x.view(n*t,c,h,w) #時間軸を一旦忘れる
                    y,z,kl_div,loss_rec=vae(x.to(DEVICE))
                    break
                for i, data in enumerate(train_vae_loader, 0):
                    x_trained, _ = data
                    y_trained,z,kl_div,loss_rec=vae(x_trained.to(DEVICE))
                    break
            show_images(
                x.to("cpu").detach().numpy(),
                y.to("cpu").detach().numpy(),
                is_show=False,save_file_name=result_dir/f"test_reconst_epoch{epoch+1}.png"
            )
            show_images(
                x_trained.to("cpu").detach().numpy(),
                y_trained.to("cpu").detach().numpy(),
                is_show=False,save_file_name=result_dir/f"trained_reconst_epoch{epoch+1}.png"
            )
    #>> VAEの学習 >>
            
    
    #>> Inceptionの学習 >>
    inception=Inception1D(
        in_channels=conf["Inception"]["latent_dim"],
        num_classes=conf["Inception"]["num_class"],
    )
    optimizer=torch.optim.Adam(params=inception.parameters(),lr=conf["Inception"]["lr"])
    optimizer.param_groups[0]["caputurable"]=True
    criterion=torch.nn.CrossEntropyLoss()
    epochs=conf["Inception"]["epochs"]
    loss_db=pd.DataFrame([])

    vae.eval() #vaeはもう学習しない
    for epoch in range(epochs):
        train_loss=0
        inception.train()
        for i,data in enumerate(train_inception_loader):
            x,label=data
            n,t,c,h,w=x.shape
            x=x.view(n*t,c,h,w) #時間軸を一旦忘れる
            _,z,_,_=vae(x.to(DEVICE)) #vaeにより特徴に変換

            z:torch.Tensor=z.view(n,t,-1) #時間軸の復元
            z=z.permute(0,2,1) #特徴軸をチャンネルに持っていく
            y=inception(z) #クラス予測

            loss:torch.Tensor=criterion(y,label.type(torch.long))
            loss.backward()
            train_loss+=loss.item()

            optimizer.step()
            optimizer.zero_grad()

            loss_db=register_dictloss(
                loss_db,epoch,i,
                {"loss":loss.item()}
            )

        if (epoch+1)%conf["save_interval"]==0:
            print(f'[{epoch + 1}] loss: {loss:.3f}')

            loss_db.to_csv(
                result_dir/"loss_inception.csv",index=False
            )
            plot_loss(
                loss_db,save_file_name=result_dir/"loss_inception_trj.png"
            )

            vae.eval()  # モデルを評価モードに設定
            with torch.no_grad():
                inception.eval()
                for i, data in enumerate(test_loader, 0):
                    x,label=data
                    n,t,c,h,w=x.shape
                    x=x.view(n*t,c,h,w) #時間軸を一旦忘れる
                    _,z,_,_=vae(x.to(DEVICE)) #vaeにより特徴に変換

                    z:torch.Tensor=z.view(n,t,-1) #時間軸の復元
                    z=z.permute(0,2,1) #特徴軸をチャンネルに持っていく
                    y=inception(z) #クラス予測
                    break

            x=x.view(n,t,c,h,w) #時間軸復元
            classes=[i for i in range(10)]

            # プロットの初期化
            n=5 #描画する要素数
            fig, ax = plt.subplots(2, n, figsize=(12,6))

            # 各要素ごとに正解ラベルと予測確率を描画
            for idx in range((n)):
                # 入力画像の描画
                ax[0][idx].imshow(x[idx][-1][0].to("cpu").detach().numpy())  # 入力画像を表示
                ax[0][idx].set_title(f'rue Label: {classes[int(label[idx].item())]}')
                ax[0][idx].axis('off')

                # 予測確率の描画
                probs = y[idx].softmax(dim=0)  # ソフトマックス関数を適用して確率に変換
                ax[1][idx].barh(classes, probs.cpu().detach().numpy())  # 各クラスの確率を水平棒グラフで表示
                ax[1][idx].set_xlabel('Probability')
                ax[1][idx].set_xlim(0, 1)

            fig.savefig(result_dir/f"estimate_class_epoch{epoch+1}.png")
    #>> Inceptionの学習 >>
        
        
if __name__=="__main__":
    main()

