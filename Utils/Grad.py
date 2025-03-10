# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:12:52 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:12:52 
#  */
import torch 
import torch.nn.functional as F

def FDM_2d(u:torch.tensor, dx_or_meshx, dy_or_meshy):
    '''Compute 1st deritive of u with FDM
    Input: 
        u: size(batch_size, my_size, mx_size, 1)
    Return:
        dudx: size(batch_size, my_size-2, mx_size-2, 1)
        dudy: size(batch_size, my_size-2, mx_size-2, 1)
    '''
    if type(dx_or_meshx)!=float or type(dy_or_meshy)!=float:
        # meshx, meshy: size(batch_size, 1, my_size, mx_size)
        deltax = (dx_or_meshx[:, 1:-1, 2:, :] - dx_or_meshx[:, 1:-1, :-2, :])
        deltay = (dy_or_meshy[:, 2:, 1:-1, :] - dy_or_meshy[:, :-2, 1:-1, :])
    else:
        deltax = 2. * dx_or_meshx
        deltay = 2. * dy_or_meshy
    #
    dudx = (u[:, 1:-1, 2:, :] - u[:, 1:-1, :-2, :]) / deltax
    dudy = (u[:, 2:, 1:-1, :] - u[:, :-2, 1:-1, :]) / deltay

    return dudx, dudy

def du_FDM_2d(u:torch.tensor, deltax:float, dim:int, 
       order:int=1, padding:str='zeros'):
    '''Compute 1st deritive of u with FDM (be careful with the size of u)
    @@@@: Require equal distance meshgrids!!!
    Input:
        u_mesh: size(batch_size, my_size, mx_size, 1)
        deltax: x(i+1) - x(i)
        dim: deritivate w.r.t x-axis or y-axis
        order: (x(i+1) - x(i-1))/deltax or ...
    Return:
        output: size(batch_size, my_size, mx_size, 1)
    '''
    assert dim==0 or dim==1
    u = u.permute(0, 3, 1, 2)
    # get weights in FDM 
    if order ==1:
        ddx1D = torch.Tensor([-0.5, 0., 0.5]).to(u.device)
    elif order==3:
        ddx1D = torch.Tensor([-1./60, 3./20, -3./4, 0., 3./4, -3./20, 1./60]).to(u.device)
    else:
        raise NotImplementedError(f'order={order} is not available')
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + (1-dim) * [1] + [-1] + dim*[1])
    # Padding the u
    if padding == "zeros":
        u = F.pad(u, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "copy":
        u = F.pad(u, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    else:
        raise NotImplementedError(f'padding={padding} is not available')
    # calculate derative w.r.t dim (using conv2d method)
    output = F.conv2d(u, ddx3D, padding="valid")
    output = output / deltax
    # remove the padding
    if dim==0:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]
    else:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]

    return output.permute(0, 2, 3, 1)

def ddu_FDM_2d(u:torch.tensor, deltax:float, dim:int, 
        order:int=1, padding:str='zeros'):
    '''Compute 2nd deritive of u with FDM (be careful with the size of u)
    @@@@: Require equal distance meshgrids!!!
    Input:
        u: size(batch_size, my_size, mx_size, 1)
        deltax: x(i+1) - x(i)
        dim: deritivate w.r.t x-axis or y-axis
        order: (x(i+1) - x(i-1))/deltax or ...
    Return:
        output: size(batch_size, my_size, mx_size, 1)
    '''
    assert dim==0 or dim==1
    u = u.permute(0, 3, 1, 2)
    # get weights in the FDM 
    if order ==1:
        ddx1D = torch.Tensor([1., -2., 1.]).to(u.device)
    elif order==3:
        ddx1D = torch.Tensor([1./90, -3./20, 3./2, -49./18, 
                              3./2, -3./20, 1./90]).to(u.device)
    else:
        raise NotImplementedError(f'order={order} is not available')
    ddx3D = torch.reshape(ddx1D, shape=[1, 1] + (1-dim) * [1] + [-1] + dim*[1])
    # padding the u
    if padding == "zeros":
        u = F.pad(u, 4 * [(ddx1D.shape[0] - 1) // 2], "constant", 0)
    elif padding == "copy":
        u = F.pad(u, 4 * [(ddx1D.shape[0] - 1) // 2], "replicate")
    else:
        raise NotImplementedError(f'padding={padding} is not available')
    # calculate derative w.r.t dim (using conv2d method)
    output = F.conv2d(u, ddx3D, padding="valid")
    output = output / deltax**2
    # remove the padding
    if dim==0:
        output = output[:, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2, :]
    else:
        output = output[:, :, :, (ddx1D.shape[0] - 1) // 2 : -(ddx1D.shape[0] - 1) // 2]

    return output.permute(0, 2, 3, 1)
    