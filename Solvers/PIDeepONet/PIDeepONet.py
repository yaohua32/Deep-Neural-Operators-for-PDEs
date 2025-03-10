# /*
#  * @Author: yaohua.zang 
#  * @Date: 2024-09-04 09:50:42 
#  * @Last Modified by:   yaohua.zang 
#  * @Last Modified time: 2024-09-04 09:50:42 
#  */
import torch 
import time 
import os
import scipy.io
from tqdm import trange
#
from Solvers import Module
from Networks.DeepONets import DeepONetBatch
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
    
    def getModel(self, layers_branch:list=None, layers_trunk:list=None, 
                 activation_branch:str=None, activation_trunk:str=None,
                 multi_ouput_strategy:str=None, num_output:int=1,
                 netType:str='DeepONetBatch', **kwrds):
        '''Get the neural network model
        '''
        if netType=='DeepONetBatch':
            model = DeepONetBatch(
                num_output=num_output, layers_branch=layers_branch, layers_trunk=layers_trunk, 
                activation_branch=activation_branch, activation_trunk=activation_trunk, 
                multi_output_strategy=multi_ouput_strategy, device=self.device)
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
    
    def train(self, LossClass:Module.LossClass, 
              a_train, u_train, x_train, 
              a_test, u_test, x_test,
              w_data:float=1., w_pde:float=1., 
              batch_size:int=100, epochs:int=1,
              epoch_show:int=10, **kwrds):
        '''Train the model
        '''
        ############# The training and testing data
        train_loader = self.dataloader(a_train, u_train, x_train, 
                                       batch_size=batch_size, shuffle=False)
        ############# The training process
        for epoch in trange(epochs):
            loss_train_sum, loss_data_sum, loss_pde_sum = 0., 0., 0.
            for a, u, x in train_loader:
                lossClass = LossClass(self)
                ############# Calculate losses
                a, u, x = a.to(self.device), u.to(self.device), x.to(self.device)
                #
                loss_pde = lossClass.Loss_pde(a, w_pde)
                loss_data = lossClass.Loss_data(x, a, u, w_data)
                loss_train = w_data * loss_data + w_pde * loss_pde
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
            a, u, x = a_test.to(self.device), u_test.to(self.device), x_test.to(self.device)
            lossClass = LossClass(self)
            try: # when no gradients are required 
                with torch.no_grad():
                    loss_test = lossClass.Loss_data(x, a, u, w_data=1.)
                    error_test = lossClass.Error(x, a, u)
            except: # when gradients are required
                loss_test = lossClass.Loss_data(x, a, u, w_data=1.)
                error_test = lossClass.Error(x, a, u)
            #
            self.callBack(loss_train_sum/len(train_loader), loss_data_sum/len(train_loader), 
                          loss_pde_sum/len(train_loader), loss_test, error_test, self.t_start)
            #
            if isinstance(error_test, list):
                error_test = sum(error_test)/len(error_test)
            if error_test.item()<self.best_err_test:
                self.best_err_test = error_test.item()
                self.saveModel(kwrds['save_path'], 'model_pideeponet_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type=='Plateau':
                self.scheduler.step(error_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss:{loss_train_sum.item()/len(train_loader):.4f}, loss_pde:{loss_pde_sum.item()/len(train_loader):.4f}, loss_data:{loss_data_sum.item()/len(train_loader):.4f}')
                for para in self.optimizer.param_groups:
                    print(f"                l2_test:{error_test.item():.4f}, lr:{para['lr']}")
        ########################
        self.saveModel(kwrds['save_path'], name='model_pideeponet_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_pideeponet')
        print(f'The total training time is {time.time()-self.t_start:.4f}')

    def train_index(self, LossClass:Module.LossClass, 
                    a_train, u_train, x_train, 
                    a_test, u_test, x_test,
                    w_data:float=1., w_pde:float=1., 
                    batch_size:int=100, epochs:int=1,
                    epoch_show:int=10, **kwrds):
        '''Train the model
        '''
        assert u_train.shape[0]==a_train.shape[0]
        assert x_train.shape[0]==a_train.shape[0]
        ############# The training and testing data
        index_loader = self.indexloader(a_train.shape[0],
                                        batch_size=batch_size, shuffle=False)
        ############# The training process
        for epoch in trange(epochs):
            loss_train_sum, loss_data_sum, loss_pde_sum = 0., 0., 0.
            for index in index_loader:
                lossClass = LossClass(self)
                ############# Calculate losses
                loss_pde = lossClass.Loss_pde(index, w_pde)
                loss_data = lossClass.Loss_data(index, w_data)
                loss_train = w_data * loss_data + w_pde * loss_pde
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
            a, u, x = a_test.to(self.device), u_test.to(self.device), x_test.to(self.device)
            lossClass = LossClass(self)
            try: # when no gradients are required 
                with torch.no_grad():
                    loss_test = lossClass.Loss_data(torch.concat([index for index in index_loader], dim=0), 
                                                    w_data=1.)
                    error_test = lossClass.Error(x, a, u)
            except: # when gradients are required
                loss_test = lossClass.Loss_data(torch.concat([index for index in index_loader], dim=0), 
                                                w_data=1.)
                error_test = lossClass.Error(x, a, u)
            #
            self.callBack(loss_train_sum/len(index_loader), loss_data_sum/len(index_loader), 
                          loss_pde_sum/len(index_loader), loss_test, error_test, self.t_start)
            #
            if isinstance(error_test, list):
                error_test = sum(error_test)/len(error_test)
            if error_test.item()<self.best_err_test:
                self.best_err_test = error_test.item()
                self.saveModel(kwrds['save_path'], 'model_pideeponet_besterror', self.model_dict)
            #######################  The scheduler
            if self.scheduler_type is None:
                pass
            elif self.scheduler_type=='Plateau':
                self.scheduler.step(error_test.item())
            else:
                self.scheduler.step()
            #######################
            if (epoch+1)%epoch_show==0:
                print(f'Epoch:{epoch+1} Time:{time.time()-self.t_start:.4f}, loss:{loss_train_sum.item()/len(index_loader):.4f}, loss_pde:{loss_pde_sum.item()/len(index_loader):.4f}, loss_data:{loss_data_sum.item()/len(index_loader):.4f}')
                for para in self.optimizer.param_groups:
                    print(f"                l2_test:{error_test.item():.4f}, lr:{para['lr']}")
        ########################
        self.saveModel(kwrds['save_path'], name='model_pideeponet_final', 
                       model_dict=self.model_dict)
        self.saveLoss(kwrds['save_path'], name='loss_pideeponet')
        print(f'The total training time is {time.time()-self.t_start:.4f}')