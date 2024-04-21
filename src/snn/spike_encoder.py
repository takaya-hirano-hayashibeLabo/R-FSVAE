import torch

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
            torch.rand(size=x.shape)<=x,
            1,
            0
        ) for _ in range(self.snn_time_step)]
        out=torch.stack(out).type(torch.float)

        return out
