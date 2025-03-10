import torch 
import torch.nn as nn
try:
    from FunActivation import FunActivation
except:
    from .FunActivation import FunActivation

class SpectralConv1d(nn.Module):

    def __init__(self, in_size:int, out_size:int, modes:int, dtype):
        super(SpectralConv1d, self).__init__()
        '''1D Fourier layer: FFT -> linear transform -> Inverse FFT
        '''
        self.in_size = in_size 
        self.out_size = out_size 
        self.modes = modes
        #
        self.scale = 1./(in_size * out_size)
        #
        if (dtype is None) or (dtype==torch.float32):
            ctype = torch.complex64 
        elif (dtype==torch.float64):
            ctype = torch.complex128 
        else:
            raise TypeError(f'No such data type.')
        #
        self.weight = nn.Parameter(self.scale * torch.rand(in_size, out_size, 
                                                           self.modes, dtype=ctype))

    def compl_mul_1d(self, input, weights):
        '''Complex multiplication: (batch_size, in_size, m) , (in_size, out_size, m) -> (batch_size, out_size, m)
        '''
        return torch.einsum("bim, iom->bom", input, weights)

    def forward(self, x):
        '''
        Input: 
            x: size(batch_size, in_size, mesh_size)
        '''
        batch_size = x.shape[0]
        ######## Compute Fourier coefficients up to factor of e^{-c}
        x_ft = torch.fft.rfft(x)  
        ######## Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_size, x.size(-1)//2+1, 
                             device=x.device, dtype=torch.cfloat)
        out_ft[:,:,:self.modes] = self.compl_mul_1d(x_ft[:,:,:self.modes], self.weight)
        ######## Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1)) 

        return x

class FNO1d(nn.Module):

    def __init__(self, in_size:int, out_size:int, modes:int, hidden_list:list[int], 
                 activation:str='ReLU', dtype=None):
        super(FNO1d, self).__init__()
        # Activation
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation
        # The input layer: 
        self.fc_in = nn.Linear(in_size, hidden_list[0], dtype=dtype)
        # The hidden layer
        conv_net, w_net = [], []
        self.hidden_in = hidden_list[0]
        for hidden in hidden_list:
            conv_net.append(SpectralConv1d(self.hidden_in, hidden, modes, dtype))
            w_net.append(nn.Conv1d(self.hidden_in, hidden, 1, dtype=dtype))
            self.hidden_in =  hidden 
        self.spectral_conv = nn.Sequential(*conv_net)
        self.weight_conv = nn.Sequential(*w_net)
        # The output layer
        self.fc_out0 = nn.Linear(self.hidden_in, 128, dtype=dtype)
        self.fc_out1 = nn.Linear(128, out_size, dtype=dtype)
    
    def forward(self, x):
        '''
        Input: 
            x: size(batch_size, mesh_size, in_size)
        Output:
            x: size(batch_size, mesh_size, out_size)
        '''
        # The input layer
        x = self.fc_in(x)
        x = x.permute(0, 2, 1)
        # The spectral conv layer 
        for conv, weight in zip(self.spectral_conv, self.weight_conv):
            x1 = conv(x)
            x2 = weight(x)
            x = self.activation(x1+x2)
        # The output layer
        x = x.permute(0, 2, 1)
        x = self.fc_out0(x)
        x = self.activation(x)

        return self.fc_out1(x)

