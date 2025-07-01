# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:06:51 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:06:51 
#  */
import torch 
import torch.nn as nn 
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation

class MultiONetBatch(nn.Module):

    def __init__(self, in_size_x:int, in_size_a:int,
                 hidden_list:list[int],
                 activation_x='SiLU_Sin',
                 activation_a='SiLU_Id', dtype=None):
        super(MultiONetBatch, self).__init__()
        self.hidden_list = hidden_list
        # Activation
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        # Activation
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        # The input layer: 
        self.fc_x_in = nn.Linear(in_size_x, hidden_list[0], dtype=dtype)
        self.fc_a_in = nn.Linear(in_size_a, hidden_list[0], dtype=dtype)
        # The hidden layer
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            net_a.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in =  hidden 
        self.net_x = nn.Sequential(*net_x)
        self.net_a = nn.Sequential(*net_a)
        # The output layer
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.l)]
            )
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
    
    def forward(self, x, a_mesh):
        '''
        Input:
            x: size(n_batch, n_mesh, dx)
            a_mesh: size(n_batch, latent_size)
        '''
        assert x.shape[0]==a_mesh.shape[0]
        #### THe input layer
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        #### The conv layer
        out = 0.
        for net_x, net_a, w in zip(self.net_x, self.net_a, self.w):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += torch.einsum('bnh,bh->bn', x, a_mesh) * w 
        ##### The output layer
        out = out/len(self.net_x) + self.b

        return out.unsqueeze(-1)

#####################
class MultiONetBatch_X(nn.Module):
    '''Multi-Input&Output case'''

    def __init__(self, in_size_x:int, in_size_a:int,
                 latent_size:int, out_size:int,
                 hidden_list:list[int],
                 activation_x='SiLU_Sin',
                 activation_a='SiLU_Id', dtype=None):
        super(MultiONetBatch_X, self).__init__()
        self.hidden_list = hidden_list
        # Activation
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        # Activation
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        # The input layer: 
        self.fc_x_in = nn.Linear(in_size_x, hidden_list[0], dtype=dtype)
        self.fc_a_in = nn.Linear(in_size_a, hidden_list[0], dtype=dtype)
        # The hidden layer
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            net_a.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in =  hidden 
        self.net_x = nn.Sequential(*net_x)
        self.net_a = nn.Sequential(*net_a)
        # The output layer
        self.fc_out = nn.Linear(latent_size, out_size, dtype=dtype)
    
    def forward(self, x, a_mesh):
        '''
        Input:
            x: size(n_batch, n_mesh, dx)
            a_mesh: size(n_batch, latent_size, da)
        '''
        assert x.shape[0]==a_mesh.shape[0]
        #### The input layer
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        #### The conv layer
        out = 0.
        for net_x, net_a in zip(self.net_x, self.net_a):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += torch.einsum('bnh,bmh->bnm', x, a_mesh)
        ##### The output layer
        out = self.fc_out(out/len(self.net_x))

        return out

############################################ Cartesian types
class MultiONetCartesianProd(nn.Module):

    def __init__(self, in_size_x:int, in_size_a:int, 
                 hidden_list:list[int],
                 activation_x='SiLU_Sin',
                 activation_a='SiLU_Id', dtype=None):
        super(MultiONetCartesianProd, self).__init__()
        self.hidden_list = hidden_list
        # Activation
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        # Activation
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        # The input layer: 
        self.fc_x_in = nn.Linear(in_size_x, hidden_list[0], dtype=dtype)
        self.fc_a_in = nn.Linear(in_size_a, hidden_list[0], dtype=dtype)
        # The hidden layer
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            net_a.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in =  hidden 
        self.net_x = nn.Sequential(*net_x)
        self.net_a = nn.Sequential(*net_a)
        # The output layer 
        self.w = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.l)]
            )
        self.b = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
    
    def forward(self, x, a_mesh):
        '''
        Input:
            x: size(mesh_size, dx)
            a_mesh: size(n_batch, latent_size)
        '''
        #### THe input layer
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        #### The conv layer
        out = 0.
        for net_x, net_a, w in zip(self.net_x, self.net_a, self.w):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += torch.einsum('bh,mh->bm', a_mesh, x) * w 
        ##### The output layer
        out = out/len(self.net_x) + self.b

        return out.unsqueeze(-1)

#############################
class MultiONetCartesianProd_X(nn.Module):
    '''Multi-Input&Output case'''

    def __init__(self, in_size_x:int, in_size_a:int, 
                 latent_size:int, out_size:int,
                 hidden_list:list[int],
                 activation_x='SiLU_Sin',
                 activation_a='SiLU_Id', dtype=None):
        super(MultiONetCartesianProd_X, self).__init__()
        self.hidden_list = hidden_list
        # Activation
        if isinstance(activation_x, str):
            self.activation_x = FunActivation()(activation_x)
        else:
            self.activation_x = activation_x
        # Activation
        if isinstance(activation_a, str):
            self.activation_a = FunActivation()(activation_a)
        else:
            self.activation_a = activation_a
        # The input layer: 
        self.fc_x_in = nn.Linear(in_size_x, hidden_list[0], dtype=dtype)
        self.fc_a_in = nn.Linear(in_size_a, hidden_list[0], dtype=dtype)
        # The hidden layer
        net_x, net_a = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            net_x.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            net_a.append(nn.Linear(self.hidden_in, hidden, dtype=dtype))
            self.hidden_in =  hidden 
        self.net_x = nn.Sequential(*net_x)
        self.net_a = nn.Sequential(*net_a)
        # The output layer
        self.fc_out = nn.Linear(latent_size, out_size, dtype=dtype)
    
    def forward(self, x, a_mesh):
        '''
        Input:
            x: size(mesh_size, dx)
            a_mesh: size(n_batch, latent_size, da)
        '''
        #### THe input layer
        x = self.activation_x(self.fc_x_in(x))
        a_mesh = self.activation_a(self.fc_a_in(a_mesh))
        #### The conv layer
        out = 0.
        for net_x, net_a in zip(self.net_x, self.net_a):
            a_mesh = self.activation_a(net_a(a_mesh))
            x = self.activation_x(net_x(x))
            out += torch.einsum('bmh,nh->bnm', a_mesh, x)
        ##### The output layer
        out = self.fc_out(out/len(self.net_x))

        return out