# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-08-22 14:10:56 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-08-22 14:10:56 
#  */
import torch 
import time 
import os
import scipy.io
from tqdm import trange
#
from Solvers import Module
from Networks.DeepONets import DeepONetBatch, DeepONetCartesianProd
#
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
        self.l2_test_list = []
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
        dict_loss['l2_test'] = self.l2_test_list
        dict_loss['time'] = self.time_list
        scipy.io.savemat(path+f'{name}.mat', dict_loss)

    def callBack(self, loss_train, loss_test, 
                 l2_test, t_start):
        '''call back
        '''
        self.loss_train_list.append(loss_train.item())
        self.loss_test_list.append(loss_test.item())
        self.time_list.append(time.time()-t_start)
        #
        if isinstance(l2_test, list):
            errs = [err.item() for err in l2_test]
            self.l2_test_list.append(errs)
        else:
            self.l2_test_list.append(l2_test.item())

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

    def getModel(self, layers_branch:list, layers_trunk:list, 
                 activation_branch:str, activation_trunk:str,
                 multi_ouput_strategy:str=None, num_output:int=1,
                 netType:str='Cartesian'):
        '''Get the neural network model
        '''
        if netType=='Cartesian':
            model = DeepONetCartesianProd(
                num_output=num_output, layers_branch=layers_branch, layers_trunk=layers_trunk, 
                activation_branch=activation_branch, activation_trunk=activation_trunk, 
                multi_output_strategy=multi_ouput_strategy)
        elif netType=='Batch':
            model = DeepONetBatch(
                num_output=num_output, layers_branch=layers_branch, layers_trunk=layers_trunk, 
                activation_branch=activation_branch, activation_trunk=activation_trunk, 
                multi_output_strategy=multi_ouput_strategy)
        else:
            raise NotImplementedError
        
        return model.to(self.device)

    def train_setup(self, model_dict:dict, lr:float=1e-3, 
                    optimizer='Adam', scheduler_type:str=None,
                    step_size=200, gamma=1/2, patience=20, factor=1/2):
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
        else:
            raise NotImplementedError
        # ####### Set the scheduler
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

    def train_batch(self, LossClass:Module.LossClass, 
              a_train, u_train, x_train, 
              a_test, u_test, x_test, 
              batch_size:int=100, epochs:int=1,
              epoch_show:int=50, **kwrds):
        '''Train the model
        '''
        ############# The dataset
        train_loader = self.dataloader(a_train, u_train, x_train, batch_size=batch_size)
        ############# The training process
        for epoch in trange(epochs):
            loss_train = 0.
            for a, u, x in train_loader:
                lossClass = LossClass(self)
                ############# Calculate losses and errors
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                loss = lossClass.Loss_data(a, u, x)
                #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter += 1
                #
                loss_train += loss
            ###################### The testing loss and error
            a, u, x = a_test.to(self.device), u_test.to(self.device), x_test.to(self.device)
            lossClass = LossClass(self)
            with torch.no_grad():
                loss_test = lossClass.Loss_data(a, u, x)
                l2_test = lossClass.Error(a, u, x)
            #
            self.callBack(loss_train/len(train_loader), loss_test, 
                          l2_test, self.t_start)
            #
            if l2_test.item()<self.best_err_test:
                self.best_err_test = l2_test.item()
                self.saveModel(kwrds['save_path'], 'model_deeponet_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler is None:
                pass
            elif self.scheduler=='Plateau':
                self.scheduler.step(l2_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss_train:{loss_train.item()/len(train_loader):.6f}, loss_test:{loss_test.item():.6f}')
                for para in self.optimizer.param_groups:
                    print('          lr:', para['lr'], 'err_test', l2_test.item())
        ########################
        self.saveModel(kwrds['save_path'], name='model_deeponet_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_deeponet')
        print(f'The total training time is {time.time()-self.t_start:.4f}')
    
    def train_cartesian(self, LossClass:Module.LossClass, 
              a_train, u_train, gridx_train, 
              a_test, u_test, gridx_test, 
              batch_size:int=100, epochs:int=1,
              epoch_show:int=50, **kwrds):
        '''Train the model
        '''
        ############# The dataset
        train_loader = self.dataloader(a_train, u_train, batch_size=batch_size)
        ############# The training process
        for epoch in trange(epochs):
            loss_train = 0.
            for a, u in train_loader:
                lossClass = LossClass(self)
                ############# Calculate losses and errors
                a, u = a.to(self.device), u.to(self.device)
                loss = lossClass.Loss_data(a, u, gridx_train.to(self.device))
                #
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.iter += 1
                #
                loss_train += loss
            ###################### The testing loss and error
            a, u = a_test.to(self.device), u_test.to(self.device)
            lossClass = LossClass(self)
            with torch.no_grad():
                loss_test = lossClass.Loss_data(a, u, gridx_test.to(self.device))
                l2_test = lossClass.Error(a, u, gridx_test.to(self.device))
            #
            self.callBack(loss_train/len(train_loader), loss_test, 
                          l2_test, self.t_start)
            #
            if l2_test.item()<self.best_err_test:
                self.best_err_test = l2_test.item()
                self.saveModel(kwrds['save_path'], 'model_deeponet_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler is None:
                pass
            elif self.scheduler=='Plateau':
                self.scheduler.step(l2_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss_train:{loss_train.item()/len(train_loader):.6f}, loss_test:{loss_test.item():.6f}')
                for para in self.optimizer.param_groups:
                    print('          lr:', para['lr'], 'err_test', l2_test.item())
        ########################
        self.saveModel(kwrds['save_path'], name='model_deeponet_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_deeponet')
        print(f'The total training time is {time.time()-self.t_start:.4f}')


