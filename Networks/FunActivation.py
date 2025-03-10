# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-09-05 15:19:50 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-09-05 15:19:50 
#  */
import torch.nn as nn
import torch

class Sinc(nn.Module):

    def __init__(self):
        super(Sinc, self).__init__()
    
    def forward(self, x):
        return x * torch.sin(x)

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * nn.functional.sigmoid(x)

class Tanh_Sin(nn.Module):

    def __init__(self):
        super(Tanh_Sin, self).__init__()

    def fun_sin(self, x):
        ''' '''
        return torch.sin(torch.pi * (x+1.))
    
    def forward(self, x):
        return nn.functional.tanh(self.fun_sin(x)) + x

class FunActivation:

    def __init__(self, **kwrds):
        self.activation = {
            'Identity': nn.Identity(),
            'ReLU': nn.ReLU(),
            'ELU': nn.ELU(),
            'Softplus': nn.Softplus(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'SiLU': nn.SiLU(),
            'Swish': Swish(),
            'Sinc': Sinc(),
            'Tanh_Sin': Tanh_Sin(),
            }
    
    def __call__(self, type=str):
        return self.activation[type]
