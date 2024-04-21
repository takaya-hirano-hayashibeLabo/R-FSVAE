from torch import nn
import torch

from .spike_encoder import SpikeEncoder
from .fsvae import FSVAE
from . import global_v as glv

from snntorch import functional as SF
from snntorch import utils
import snntorch as snn
from snntorch import surrogate


class RecogFSVAE(nn.Module):
    """
    分類とFSVAEの学習を一辺にやらせる
    これによって, 分布をくっきり分けるのが目的
    """

    def __init__(self,spike_encoder:SpikeEncoder,fsvae_hiddens,recog_hiddens,device):
        super(RecogFSVAE,self).__init__()

        self.spike_enc=spike_encoder
        self.fsvae=FSVAE(hidden_dims=fsvae_hiddens)


        #>> 分類モデルの構築 >>
        modules=[]
        spike_grad = surrogate.fast_sigmoid(slope=25)
        beta=0.5
        threshold=1.0

        #入力層
        modules+=[ 
            nn.Linear(glv.network_config['latent_dim'],fsvae_hiddens[0]),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
        ]

        # 中間層
        prev_dim=fsvae_hiddens[0]
        for hidden_dim in  recog_hiddens:
            modules+=[
                nn.Linear(prev_dim,hidden_dim),
                snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold),
            ]
            prev_dim=hidden_dim

        # 出力層
        modules+=[ 
            nn.Linear(prev_dim,10),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True,threshold=threshold,output=True),
        ]
        self.recog_model=nn.Sequential(*modules) #分類モデル
        self.criterion_recog=SF.ce_rate_loss()
        #>> 分類モデルの構築 >>

        self.device=device


    def forward(self,x:torch.Tensor):
        """
        :param x : [batch x c x h x w]
        :return x_recon
        :return q_z
        :return p_z
        :return z
        :return pred_spikes
        """

        in_spike=self.spike_enc(x.to(self.device)) #スパイク変換

        # >> fsvaeによる再構成 >>
        x_recon,q_z,p_z,z=self.fsvae.forward(
            in_spike.permute(1,2,3,4,0),
            scheduled=glv.network_config['scheduled']
        )
        # >> fsvaeによる再構成 >>

        #>> 分類モデルによる予測 >>
        utils.reset(self.recog_model)
        out_spikes=[]
        for t_snn in range(glv.network_config['n_steps']):
            out_spike,_=self.recog_model.forward(z[:,:,t_snn])
            out_spikes+=[out_spike]
        pred_spikes=torch.stack(out_spikes).to(float)
        #>> 分類モデルによる予測 >>

        return x_recon,q_z,p_z,z,pred_spikes
    

    def loss_func(self,x,x_recon,q_z,p_z,pred_spikes,label):
        """
        :param x : [batch x c x h x w]
        :param x_recon
        :param q_z
        :param p_z
        :param pred_spikes
        :param label: 正解ラベル
        :return loss_recons
        :return loss_distance
        :return loss_pred
        """

        #>> 再構成loss >>
        fsvae_loss=self.fsvae.loss_function_mmd(
            x.to(self.device),x_recon,q_z,p_z
        )
        loss_recons, loss_distance=fsvae_loss["Reconstruction_Loss"],fsvae_loss['Distance_Loss']
        #>> 再構成loss >>

        #>> 予測loss >>
        loss_pred=self.criterion_recog(pred_spikes,label.to(int).to(self.device))
        #>> 予測loss >>

        return loss_recons,loss_distance,loss_pred
