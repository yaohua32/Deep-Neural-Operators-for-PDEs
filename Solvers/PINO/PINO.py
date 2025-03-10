# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:11:45 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:11:45 
#  */
import torch 
import time 
import os
import scipy.io
from tqdm import trange
#
from Solvers import Module
from Networks.FNO1d import FNO1d
from Networks.FNO2d import FNO2d
#
from Utils.RBFInterpolatorMesh import RBFInterpolator
from Utils.Losses import MyError, MyLoss

##########################
class Solver(Module.Solver):

    def __init__(self, device='cuda:0',
                 dtype=torch.float32):
        self.device = device 
        self.dtype = dtype
        #
        self.iter = 0
        self.time_list = []
        self.loss_train_list = []
        self.loss_test_list = []
        #
        self.loss_data_list = []
        self.loss_pde_list = []
        #
        self.error_list = []
        #
        self.error_setup()
        self.loss_setup()
    
    def loadModel(self, path:str, name:str):
        '''Load trained model
        '''
        return torch.load(path+f'{name}.pth', map_location=self.device)

    def saveModel(self, path:str, name:str, model_dict:dict):
        '''Save trained model (the whole model)
        '''
        if not os.path.exists(path):
            os.makedirs(path)
        #
        torch.save(model_dict, path+f'{name}.pth')

    def loadLoss(self, path:str, name:str):
        '''Load saved losses
        '''
        loss_dict = scipy.io.loadmat(path+f'{name}.mat')

        return loss_dict

    def saveLoss(self, path:str, name:str):
        '''Save losses
        '''
        dict_loss = {}
        dict_loss['loss_train'] = self.loss_train_list
        dict_loss['loss_test'] = self.loss_test_list
        dict_loss['loss_data'] = self.loss_data_list 
        dict_loss['loss_pde'] = self.loss_pde_list
        dict_loss['error'] = self.error_list
        dict_loss['time'] = self.time_list
        scipy.io.savemat(path+f'{name}.mat', dict_loss)

    def callBack(self, loss_train, loss_data, loss_pde,
                 loss_test, error_test, t_start):
        '''call back
        '''
        self.loss_train_list.append(loss_train.item())
        self.loss_test_list.append(loss_test.item())
        self.loss_data_list.append(loss_data.item())
        self.loss_pde_list.append(loss_pde.item())
        self.time_list.append(time.time()-t_start)
        #
        if isinstance(error_test, list):
            errs = [err.item() for err in error_test]
            self.error_list.append(errs)
        else:
            self.error_list.append(error_test.item())

    def error_setup(self, err_type:str='lp_rel', d:int=2, p:int=2, 
                    size_average=True, reduction=True):
        ''' setups of error
        Input:
            err_type: from {'lp_rel', 'lp_abs'}
        '''
        Error = MyError(d, p, size_average, reduction)
        #
        if err_type=='lp_rel':
            self.getError = Error.Lp_rel
        elif err_type=='lp_abs':
            self.getError = Error.LP_abs
        else:
            raise NotImplementedError(f'{err_type} has not defined.')
    
    def loss_setup(self, loss_type:str='mse_org', 
                   size_average=True, reduction=True):
        '''setups of loss
        Input:
            loss_type: from {'mse_org', 'mse_rel'}
        '''
        Loss = MyLoss(size_average, reduction)
        #
        if loss_type=='mse_org':
            self.getLoss = Loss.mse_org
        elif loss_type=='mse_rel':
            self.getLoss = Loss.mse_rel
        else:
            raise NotImplementedError(f'{loss_type} has not defined.')

    def getModel_a(self, Exact_a:object=None, approximator:str='RBF',
                   **kwrds):
        '''The model for coefficient a
        '''
        if Exact_a is not None:
            print('Using the exact definition of a.')
            model_a = Exact_a
        elif approximator=='RBF':
            x_mesh = kwrds['x_mesh'].to(self.device)
            model_a = RBFInterpolator(
                x_mesh=x_mesh, kernel=kwrds['kernel'], 
                eps=kwrds['eps'], degree=kwrds['degree'], 
                smoothing=kwrds['smoothing'], 
                dtype=self.dtype).to(self.device)
        else:
            raise NotImplementedError(f'No such approximator: {approximator}.')
        
        return model_a

    def getModel(self, in_size, out_size, modes, hidden_list, 
                 activation='ReLU', netType:str='FNO1d'):
        '''Get the neural network model
        '''
        self.netType = netType
        if netType=='FNO1d':
            model = FNO1d(in_size, out_size, modes, hidden_list, activation)
        elif netType=='FNO2d':
            model = FNO2d(in_size, out_size, modes[0], modes[1], hidden_list, activation)
        else:
            raise NotImplementedError
        
        return model.to(self.device)

    def train_setup(self, model_dict:dict, lr:float=1e-3, 
                    optimizer='Adam', scheduler_type:str=None,
                    step_size=500, gamma=1/3, patience=20, factor=1/2):
        '''Setups for training
        '''
        self.model_dict = model_dict
        ########### The models' parameters
        param_list = []
        for model in model_dict.values():
            param_list += list(model.parameters())
        ########### Set the optimizer
        if optimizer=='Adam':
            self.optimizer = torch.optim.Adam(
                params=param_list, lr=lr, weight_decay=1e-4)
        elif optimizer=='AdamW':
            self.optimizer = torch.optim.AdamW(
                params=param_list, lr=lr, weight_decay=1e-4)
        elif optimizer=='RMSprop':
            self.optimizer = torch.optim.RMSprop(
                params=param_list, lr=lr, weight_decay=1e-4)
        else:
            raise NotImplementedError
        ############ Set the scheduler
        if scheduler_type=='StepLR':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, 
                gamma=gamma, last_epoch=-1)
        elif scheduler_type=='Plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, patience=patience)
        self.scheduler_type = scheduler_type
        #
        self.t_start = time.time()
        self.best_err_test = 1e10
    
    def train(self, LossClass:Module.LossClass,
              a_train, u_train, a_test, u_test, 
              w_data:float=1., w_pde:float=1.,
              batch_size:int=100, epochs:int=1, 
              epoch_show:int=10, **kwrds):
        '''Train the model
        Input: 
            a_train: size(N_train, mesh_size, in_channel)
            u_train: size(N_train, mesh_szie, out_size)
            a_test: size(N_test, mesh_size, in_channel)
            u_test: size(N_test, mesh_szie, out_size)
        '''
        ############# The training and testing data
        train_loader = self.dataloader(a_train, u_train, 
                                       batch_size=batch_size, shuffle=False)
        ############# The training process
        for epoch in trange(epochs):
            loss_train_sum, loss_data_sum, loss_pde_sum = 0., 0., 0.
            for a, u in train_loader:
                a, u = a.to(self.device), u.to(self.device)
                ############# Calculate losses
                lossClass = LossClass(self)
                loss_pde = lossClass.Loss_pde(a, w_pde)
                loss_data = lossClass.Loss_data(a, u, w_data)
                loss_train = w_pde * loss_pde  + w_data * loss_data
                #
                self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()
                self.iter += 1
                #
                loss_train_sum += loss_train
                loss_data_sum += loss_data
                loss_pde_sum += loss_pde
            ###################### The testing loss and error
            a, u = a_test.to(self.device), u_test.to(self.device)
            lossClass = LossClass(self)
            try: # when no gradients are required
                with torch.no_grad():
                    loss_test = lossClass.Loss_data(a, u, w_data=1.)
                    error_test = lossClass.Error(a, u)
            except:
                loss_test = lossClass.Loss_data(a, u, w_data=1.)
                error_test = lossClass.Error(a, u)
            #
            self.callBack(loss_train_sum/len(train_loader), loss_data_sum/len(train_loader), 
                          loss_pde_sum/len(train_loader), loss_test, error_test, self.t_start)
            #
            if isinstance(error_test, list):
                error_test = sum(error_test)/len(error_test)
            if error_test.item()<self.best_err_test:
                self.best_err_test = error_test.item()
                self.saveModel(kwrds['save_path'], 'model_pino_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type=='Plateau':
                self.scheduler.step(error_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss:{loss_train_sum.item()/len(train_loader):.4f}, loss_pde:{loss_pde_sum.item()/len(train_loader):.4f}, loss_pde:{loss_data_sum.item()/len(train_loader):.4f}')
                for para in self.optimizer.param_groups:
                    print(f"                l2_test:{error_test.item():.4f}, lr:{para['lr']}")
        ########################
        self.saveModel(kwrds['save_path'], name='model_pino_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_pino')
        print(f'The total training time is {time.time()-self.t_start:.4f}')
