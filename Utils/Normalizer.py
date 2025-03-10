# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:13:13 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:13:13 
#  */
import torch

class UnitGaussianNormalizer():

    def __init__(self, x, eps=1e-8):
        super(UnitGaussianNormalizer, self).__init__()
        '''Apply normaliztion to the first dimension of last axis of x
        Input:
            x: size(N, mesh_size, 1+d)
        Output:
            mean: size(mesh_szie, 1)
            std: size(mesh_size, 1)
        '''
        self.mean = torch.mean(x[...,0:1], 0)
        self.std = torch.std(x[...,0:1], 0)
        self.eps = eps
    
    def encode(self, x):
        '''
        Input:
            x: x: size(N, mesh_size, 1+d)
        '''
        d = x.shape[-1] - 1
        x_list = torch.split(x, split_size_or_sections=[1, d], dim=-1)
        x = torch.cat([(x_list[0]-self.mean) / (self.std + self.eps), x_list[1]], dim=-1)

        return x
    
    def decode(self, x, sample_idx=None):
        '''
        '''
        if sample_idx is not None:
            # x_size(batch*n, )
            if len(self.mean.shape)==len(sample_idx[0].shape):
                std = self.std[sample_idx]
                mean = self.mean[sample_idx]
            # x_size(T*batch*n, )
            if len(self.mean.shape)>len(sample_idx[0].shape):
                std = self.std[:,sample_idx] 
                mean = self.mean[:,sample_idx]
        else:
            # x_size(n,)
            std = self.std 
            mean = self.mean
        #
        d = x.shape[-1] - 1
        x_list = torch.split(x, split_size_or_sections=[1, d], dim=-1)
        x = torch.cat([x_list[0] * (std + self.eps) + mean, x_list[1]], dim=-1)

        return x

class GaussianNormalizer():

    def __init__(self, x, eps=1e-8):
        super(GaussianNormalizer, self).__init__
        '''Apply normaliztion to the first dimension of last axis of x
        Input:
            x: size(N, mesh_size, 1+d)
        Output:
            mean: size()
            std: size()
        '''
        self.mean = torch.mean(x[...,0])
        self.std = torch.std(x[...,0])
        self.eps = eps
    
    def encode(self, x):
        '''
        Input:
            x: x: size(N, mesh_size, 1+d)
        '''
        d = x.shape[-1] - 1
        x_list = torch.split(x, split_size_or_sections=[1, d], dim=-1)
        x = torch.cat([(x_list[0]-self.mean) / (self.std + self.eps), x_list[1]], dim=-1)

        return x
    
    def decode(self, x):
        '''
        Input:
            x: size(batch*n,?) or size(T*batch*n,?)
        '''
        #
        d = x.shape[-1] - 1
        x_list = torch.split(x, split_size_or_sections=[1, d], dim=-1)
        x = torch.cat([x_list[0] * (self.std + self.eps) + self.mean, x_list[1]], dim=-1)

        return x

class RangeNormalizer():

    def __init__(self, x, low=0., high=1.):
        super(RangeNormalizer, self).__init__()
        '''Apply normaliztion to the first dimension of last axis of x
        Input:
            x: size(N, mesh_size, 1+d)
        Output:
            a: size(mesh_size)
            b: size(mesh_size)
        '''
        x_min = torch.min(x[...,0:1], 0)[0].view(-1)
        x_max = torch.max(x[...,0:1], 0)[0].view(-1)
        #
        self.a = (high - low) / (x_max - x_min)
        self.b = low - self.a * x_min
    
    def encode(self, x):
        '''
        Input:
            x: x: size(N, mesh_size, 1+d)
        '''
        d = x.shape[-1] - 1
        x_list = torch.split(x, split_size_or_sections=[1, d], dim=-1)
        #
        x0_size = x_list[0].size()
        x0 = x_list[0].reshape(x0_size[0], -1)
        x0 = self.a * x0 + self.b 
        #
        x = torch.cat([x0.reshape(x0_size), x_list[1]], dim=-1)

        return x

    def decode(self, x):
        '''
        Input:
            x: size(batch*n,?) or size(T*batch*n,?)
        '''
        d = x.shape[-1] - 1
        x_list = torch.split(x, split_size_or_sections=[1, d], dim=-1)
        #
        x0_size = x_list[0].size()
        x0 = x_list[0].reshape(x0_size[0], -1)
        x0 = (x0 - self.b) / self.a 
        #
        x = torch.cat([x0.reshape(x0_size), x_list[1]], dim=-1)

        return x
