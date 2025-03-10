# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-09-23 10:48:03 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-09-23 10:48:03 
#  */
import torch.nn as nn
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation

class ResNetBlock(nn.Module):

    def __init__(self, in_size, mid_size, activation, dtype=None):
        super(ResNetBlock, self).__init__()
        net = []
        # The first layer
        net.append(nn.Linear(in_size, mid_size, dtype=dtype))
        net.append(activation)
        # The second layer 
        net.append(nn.Linear(mid_size, mid_size, dtype=dtype))
        net.append(activation)
        # The shortcut connection
        if in_size != mid_size:
            self.shortcut = nn.Sequential(
                nn.Linear(in_size, mid_size, dtype=dtype))
        else:
            self.shortcut = nn.Sequential()
        #
        self.layer = nn.Sequential(*net)

    def forward(self, x):
        out = self.layer(x) + self.shortcut(x)

        return out

class ResNet(nn.Module):

    def __init__(self, layers_list, activation='Tanh', dtype=None):
        super(ResNet, self).__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # Network Sequential
        net = []
        self.hidden_in = layers_list[0]
        for hidden in layers_list[1:-1]:
            net.append(ResNetBlock(self.hidden_in, hidden, self.activation, dtype=dtype))
            net.append(self.activation)
            self.hidden_in = hidden
        net.append(nn.Linear(self.hidden_in, layers_list[-1], dtype=dtype))
        self.net = nn.Sequential(*net)
    
    def forward(self, x):
        x = self.net(x)

        return x