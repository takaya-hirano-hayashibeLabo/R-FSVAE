import pandas as pd
import torch
from torch.nn import functional as F
import yaml
import matplotlib.pyplot as plt
import numpy as np

def load_conf(yaml_file):
    """
    yml形式のconfigをdictにロードする
    """
    with open(yaml_file, 'r') as file:
        try:
            yaml_dict = yaml.safe_load(file)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)

def register_loss(loss_db,epoch,iter,total_loss,reconst_loss,distance):
    """
    :param loss_db:既に集めているロスのテーブル
    """

    new_loss_db=pd.DataFrame(
        [[epoch,iter,total_loss,reconst_loss,distance]],
        columns=["epoch","iter","loss","reconst","distance"]
    )
    new_loss_db=pd.concat([loss_db,new_loss_db])
    new_loss_db.reset_index(inplace=True,drop=True)

    return new_loss_db


def register_dictloss(loss_db:pd.DataFrame,epoch,iter,loss_dict:dict):
    """
    :param loss_db:既に集めているロスのテーブル
    :param loss_dict: 辞書形式のloss. 何が入っていても良い
    """

    new_data,columns=[epoch,iter],["epoch","iter"]
    for key,val in loss_dict.items():
        new_data+=[val]
        columns+=[key]

    new_loss_db=pd.DataFrame(
        [new_data],
        columns=columns
    )
    new_loss_db=pd.concat([loss_db,new_loss_db])
    new_loss_db.reset_index(inplace=True,drop=True)

    return new_loss_db


def register_dictdata(data_db,epoch,iter,data_dict:dict):
    """
    :param data_db:既に集めているロスのテーブル
    :param data_dict: 辞書形式のdata. 何が入っていても良い
    """

    new_data,columns=[epoch,iter],["epoch","iter"]
    for key,val in data_dict.items():
        new_data+=[val]
        columns+=[key]

    new_data_db=pd.DataFrame(
        [new_data],
        columns=columns
    )
    new_data_db=pd.concat([data_db,new_data_db])
    new_data_db.reset_index(inplace=True,drop=True)

    return new_data_db


def show_pred_result(x,recons,spike_counts,label,file_name,n=5):
    """
    :param x 入力画像
    :param recons 再構成画像
    :param spike_counts 発火数
    :param label 正解ラベル
    :param file_name
    :param n 描画数
    """

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
        probs = spike_counts[idx].flatten().softmax(dim=0)  # ソフトマックス関数を適用して確率に変換
        # print(y[idx])
        # print(probs)
        bars=ax[2][idx].barh(classes, probs.cpu().detach().numpy())  # 各クラスの確率を水平棒グラフで表示
        ax[2][idx].set_xlabel('Probability')
        ax[2][idx].set_xlim(0, 1)
    
        # barの先端に発火数をテキストとして表示
        for bar, prob, spike_rate in zip(bars, probs,list(spike_counts[idx].to("cpu").detach().numpy().flatten())):
            ax[2][idx].text(prob+0.15, bar.get_y() + bar.get_height()/2, f'{int(spike_rate)}', ha='right', va='center')

    fig.savefig(file_name)
    plt.close()



class DataTransformNrm():
    """
    ２次元データ（画像と同じ次元）をリサイズ＆正規化するクラス
    """
        
    def __call__(self,data,size=(28,28),max=None,min=None):
        """
        :param data: [N x C x H x W]
        :param size: 変換後のサイズ
        :return data_nrm, max, min
        """
        
        if not torch.is_tensor(data):
            data=torch.Tensor(data)
        
        if max is None and min is None:
            max=torch.max(data)
            min=torch.min(data)
            
        data_nrm=F.interpolate(
            torch.Tensor((data-min)/(1e-20+max)),
            size,mode='area'
        )
        
        return data_nrm,max,min

class Datasets(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x=x
        self.y=y

        self.datanum=x.shape[0]

    def __len__(self):
        return self.datanum
    
    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
