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
import pandas as pd

from VaeInception1d import VAE,register_loss

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
    ax1.plot(data.index, data['reconst'], color='orange', label='reconst',alpha=alpha)
    ax1.tick_params(axis='y', labelcolor=color)

    # 2つ目の y 軸に distance をプロット
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

    #>> データの準備 >>
    class RescaleMinMax:
        """VAEは入力が0から1じゃないといけない"""
        def __call__(self, tensor):
            # 各チャンネルごとの最大値と最小値を計算
            min_val = tensor.min()
            max_val = tensor.max()
            
            # 0から1にスケーリング
            tensor = (tensor - min_val) / (max_val - min_val)
            return tensor

    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        RescaleMinMax(),  # 最大値1, 最小値0に正規化
    ])
    trainset = torchvision.datasets.MNIST(root=Path(__file__).parent/'data', 
                                        train=True,
                                        download=True,
                                        transform=transform
                                       )
    trainloader = torch.utils.data.DataLoader(trainset,
                                                batch_size=100,
                                                shuffle=True,
                                                # num_workers=2,
                                                generator=torch.Generator(device=torch.Tensor([0,0]).device)
                                                )


    testset = torchvision.datasets.MNIST(root=Path(__file__).parent/'data', 
                                            train=False, 
                                            download=True, 
                                            transform=transform)
    testloader = torch.utils.data.DataLoader(testset, 
                                                batch_size=100,
                                                shuffle=True, 
                                                # num_workers=2,
                                                generator=torch.Generator(device=torch.Tensor([0,0]).device)
                                                )
    #>> データの準備 >>


    vae:torch.Module=VAE()
    optimizer=torch.optim.Adam(params=vae.parameters(),lr=0.001)
    optimizer.param_groups[0]["caputurable"]=True

    save_dir=Path(__file__).parent/"result"
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epoches=10
    loss_db=pd.DataFrame([])
    for epoch in range(epoches):
        vae.train()
        train_loss=0
        for i,data in enumerate(trainloader):
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

            if i % 100 == 99:  # 100ミニバッチごとに進捗を表示
                # print(f"loss_reconstraction:{loss_rec}, kl_divergence:{kl_div}")
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 100:.3f}')
                train_loss = 0.0

        loss_db.to_csv(
            save_dir/"loss.csv",index=False
        )
        plot_loss(
            loss_db,save_file_name=save_dir/"loss_trj.png"
        )

        vae.eval()  # モデルを評価モードに設定
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                x, _ = data
                y,z,kl_div,loss_rec=vae(x.to(DEVICE))
                break
        show_images(
            x.to("cpu").detach().numpy(),
            y.to("cpu").detach().numpy(),
            is_show=False,save_file_name=save_dir/f"fig_epoch{epoch}.png"
        )

        
if __name__=="__main__":
    main()

