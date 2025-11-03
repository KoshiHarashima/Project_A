from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from baseline.baseline import OptRevOneItem
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MyersonNet(nn.Module):
    '''
    MyersonNet: A Neural Network for computing optimal single item auctions
    '''
    def __init__(self, args, train_data, test_data):
        super(MyersonNet, self).__init__()
        self.args = args
        self.train_data = train_data
        self.test_data = test_data
        self.nn_build()
        
    def nn_build(self):
        num_func = self.args.num_linear_func
        num_max_units = self.args.num_max_units
        num_agent = self.args.num_agent
        
        ######
        # Initialization of the weights
        ######
        np.random.seed(self.args.seed_val)
        self.w_encode1_init = np.random.normal(size=(num_max_units, num_func, num_agent))
        self.w_encode2_init = -np.random.rand(num_max_units, num_func, num_agent) * 5.0
                            
        # Linear weights
        self.w_encode1 = nn.Parameter(torch.tensor(self.w_encode1_init.astype(np.float32)))
        # Bias weights
        self.w_encode2 = nn.Parameter(torch.tensor(self.w_encode2_init.astype(np.float32)))
        
    def nn_eval(self, x, mode='train'): 
        """Evaluate network given input x and mode ('train' or 'test')"""
        num_func = self.args.num_linear_func
        num_max_units = self.args.num_max_units
        num_agent = self.args.num_agent
        
        # Convert numpy to tensor if needed
        if isinstance(x, np.ndarray):
            x_tensor = torch.tensor(x, dtype=torch.float32)
        else:
            x_tensor = x
        
        batch_size = x_tensor.shape[0]
        
        append_dummy_mat = torch.tensor(
            np.append(np.identity(num_agent),
                     np.zeros([num_agent, 1]), 1).astype(np.float32))
        ######                    
        # Compute the input of the SPA with reserve zero unit
        ######
        w_encode1_copy = self.w_encode1.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        w_encode2_copy = self.w_encode2.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        x_copy = x_tensor.unsqueeze(1).unsqueeze(1).repeat(1, num_max_units, num_func, 1)
                    
        vv_max_units = torch.max(torch.multiply(
                        x_copy, torch.exp(w_encode1_copy)) + w_encode2_copy, dim=2)[0]
        # Compute virtual value
        vv = torch.min(vv_max_units, dim=1)[0]
        
        #####
        # Run SPA unit with reserve price 0
        #####
        
        # Compute allocation rate in SPA unit
        if mode == 'train':
            w_a = torch.tensor(np.identity(num_agent+1).astype(np.float32) * 1000)
            a_dummy = F.softmax(torch.matmul(torch.matmul(vv, append_dummy_mat), w_a), dim=-1)
        if mode == 'test':
            win_agent = torch.argmax(torch.matmul(vv, append_dummy_mat), dim=1)  # The index of agent who win the item
            a_dummy = F.one_hot(win_agent, num_agent+1).float()
        
        a = a_dummy[:, :num_agent]
        
        # Compute payment in SPA unit: weighted max of inputs
        w_p = torch.tensor((np.ones((num_agent, num_agent)) - np.identity(num_agent)).astype(np.float32))
        spa_tensor1 = vv.unsqueeze(0).repeat(num_agent, 1, 1)
        spa_tensor2 = torch.matmul(spa_tensor1, torch.diag_embed(w_p))
        p_spa = torch.transpose(torch.max(spa_tensor2, dim=2)[0], 0, 1)
        
        ## Decode the payment
        p_spa_copy = p_spa.unsqueeze(1).unsqueeze(1).repeat(1, num_max_units, num_func, 1)
        p_max_units = torch.min(torch.multiply(p_spa_copy - w_encode2_copy,
                        torch.reciprocal(torch.exp(w_encode1_copy))), dim=2)[0]
                            
        p = torch.max(p_max_units, dim=1)[0]
        
        # Compute the revenue
        revenue = torch.mean(torch.sum(torch.multiply(a, p), dim=1))
                    
        return revenue, a, vv
       
    def nn_train(self):
        """Train the network"""
        # Parse parameters
        sample_val = self.train_data
        
        num_func = self.args.num_linear_func
        num_max_units = self.args.num_max_units
        num_agent = self.args.num_agent
        batch_size = self.args.batch_size
        data_size = sample_val.shape[0]

        # Loss: -revenue
        # Optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)

        # Store weights when training
        num_recordings = int(np.round(self.args.num_iter/self.args.skip_iter)) + 1
        w_encode1_array = np.zeros((num_recordings, num_max_units, num_func, num_agent))
        w_encode2_array = np.zeros((num_recordings, num_max_units, num_func, num_agent))

        w_encode1_array[0,:,:,:] = self.w_encode1_init
        w_encode2_array[0,:,:,:] = self.w_encode2_init

        np.random.seed()
        # Iterate over self.args.num_iter iterations
        self.train()
        for i in range(self.args.num_iter):
            perm = np.random.choice(data_size, batch_size, replace=False)
            x_batch = torch.tensor(sample_val[perm, :], dtype=torch.float32, requires_grad=True)
            
            optimizer.zero_grad()
            revenue, _, _ = self.nn_eval(x_batch, mode='train')
            loss = -revenue
            loss.backward()
            optimizer.step()
                                
            if(i % self.args.skip_iter == 0):
                ind = int(np.round(i/self.args.skip_iter)) + 1
                w_encode1_array[ind,:,:,:] = self.w_encode1.detach().cpu().numpy()
                w_encode2_array[ind,:,:,:] = self.w_encode2.detach().cpu().numpy()
                            
            if((i+1) % 10000 == 0):
                print('Complete ' + str(i+1) + ' iterations')                        
        
        return w_encode1_array, w_encode2_array


    def nn_test(self, data, mechanism):
        """Test the network"""
        # Parse parameters
        sample_val = data
        w_encode1_array, w_encode2_array = mechanism
        num_agent = self.args.num_agent
        data_size = sample_val.shape[0]
        
        win_index = OptRevOneItem(self.args, data).winner()
        
        num_recordings = w_encode1_array.shape[0]
        rev_array = np.zeros(num_recordings)
        alloc_error_array = np.zeros(num_recordings)
        vv_array = np.zeros((data_size, num_agent))

        self.eval()
        with torch.no_grad():
            x_test = torch.tensor(sample_val, dtype=torch.float32)
            
            for i in range(num_recordings):
                # Set weights temporarily
                self.w_encode1.data = torch.tensor(w_encode1_array[i,:,:,:].astype(np.float32))
                self.w_encode2.data = torch.tensor(w_encode2_array[i,:,:,:].astype(np.float32))
                
                revenue, alloc, vv = self.nn_eval(x_test, mode='test')
                
                # Compute allocation error
                alloc_np = alloc.cpu().numpy()
                a_error = np.sum(np.abs(alloc_np - win_index)) / data_size / 2.0
                
                rev_array[i] = revenue.item()
                alloc_error_array[i] = a_error
                if i == num_recordings - 1:  # Store vv only for last recording
                    vv_array = vv.cpu().numpy()
                                   
        return (rev_array, alloc_error_array, vv_array)
