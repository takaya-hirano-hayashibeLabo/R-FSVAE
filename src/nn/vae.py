import torch
from torch import nn
from copy import deepcopy


class Encoder(nn.Module):
    def __init__(self,in_channel,channels ,latent_dim=64, pool_sizes=[2,2,2,2]):
        """
        CNNで可変サイズにすると実装がめんどいので, 入力と特徴次元は固定する.
        入力は1x32x32, 出力は16x4x4
        """
        super(Encoder,self).__init__()

        lays=[]
        prev_channel=in_channel
        # channels=[4,8,16,32]
        for i,channel in enumerate(channels):
            lays+=[
                nn.Conv2d(prev_channel, channel, kernel_size=3,stride=1,padding=1),
                nn.BatchNorm2d(channel,eps=1e-5,),
                nn.AvgPool2d(pool_sizes[i]),
                nn.ReLU(),
            ]
            prev_channel=channel
        lays.append(
            nn.Flatten()
        )
        self.conv_net=nn.Sequential(*lays)
        
        self.mean=nn.Linear(channels[-1]*2*2,latent_dim)
        self.log_var=nn.Linear(channels[-1]*2*2,latent_dim)


    def forward(self,x):
        
        x=self.conv_net(x)
        mean,log_var=self.mean(x),self.log_var(x)

        return mean,log_var
    

class Decoder(nn.Module):
    def __init__(self,enc_channel,enc_size,channels,latent_dim=64,pool_sizes=[2,2,2,2]) -> None:
        super(Decoder,self).__init__()

        param_convtrans={
            "pool2":{
                "kernel":3,"stride":2,"padding":1,"out_padding":1
            },
            "pool4":{
                "kernel":5,"stride":4,"padding":1,"out_padding":1
            }
        }

        self.enc_channel=enc_channel
        self.enc_size=enc_size

        self.linear=nn.Sequential(
            nn.Linear(latent_dim,enc_channel*enc_size*enc_size),
            nn.ReLU(),
        )

        lays=[]
        prev_channel=enc_channel
        for i,channel in enumerate(channels):
            param=param_convtrans[f"pool{pool_sizes[i]}"]
            lays+=[
                nn.ConvTranspose2d(
                    prev_channel,channel,
                    kernel_size=param["kernel"],stride=param["stride"],
                    padding=param["padding"],output_padding=param["out_padding"]
                    ),
                nn.BatchNorm2d(channel,eps=1e-5,),
            ]
            if not (i+1)==len(channels):
                lays+=[nn.ReLU()]
            else:
                lays+=[nn.Sigmoid()]
            prev_channel=channel
        self.conv_net=nn.Sequential(*lays)


    def forward(self,x):
        x=self.linear(x)
        x=torch.reshape(x,(x.shape[0],self.enc_channel,self.enc_size,self.enc_size)) #次元管理
        x=self.conv_net(x)
        return x
    

class VAE(nn.Module):
    def __init__(self,in_channel=1,channels=[4,8,16,32,64],latent_dim=256,pool_sizes=[2,2,2,2]):
        super(VAE,self).__init__()

        self.enc=Encoder(in_channel,channels,latent_dim,pool_sizes)
        channels_rev:list=deepcopy(channels)
        channels_rev.reverse()
        channels_rev=deepcopy(channels_rev[1:])+[in_channel]

        self.dec=Decoder(
            enc_channel=channels[-1],
            enc_size=2,
            channels=channels_rev,
            latent_dim=latent_dim,
            pool_sizes=pool_sizes
            )

    def sample_z(self,mean:torch.Tensor,log_var:torch.Tensor):
        """
        平均と分散からサンプリングする関数
        """
        epsilon=torch.randn(size=mean.shape)
        return mean+epsilon*torch.exp(0.5*log_var) #μ+N(0,σ)

    def forward(self,x):
        """
        :return y
        :return z
        :return kl_div
        :return loss_rec
        """
        mean,log_var=self.enc(x)
        z=self.sample_z(mean,log_var)
        y=self.dec(z)
        kl_div=-0.5 * torch.mean(1 + log_var - mean**2 - torch.exp(log_var))
        loss_rec=-torch.mean(x * torch.log(y + 1e-15) + (1 - x) * torch.log(1 - y + 1e-15))

        return y,z,kl_div,loss_rec
