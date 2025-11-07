from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Trainer(object):

    def __init__(self, config, mode, net, clip_op_lambda):
        self.config = config
        self.mode = mode
        
        # Create output-dir
        if not os.path.exists(self.config.dir_name): os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(self.config.test.num_misreports) + "_gd_" + str(self.config.test.gd_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")
            
        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        torch.manual_seed(self.config[self.mode].seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config[self.mode].seed)
        
        # Init Logger
        self.init_logger()

        # Init Net
        self.net = net
        
        ## Clip Op
        self.clip_op_lambda = clip_op_lambda
        
        # Init graph components (no TF graph, just setup optimizers and variables)
        self.init_graph()
              
    def init_logger(self):

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.FileHandler(self.log_fname, 'w')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.logger = logger

    def compute_rev(self, pay):
        """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
        return torch.mean(torch.sum(pay, dim=-1))

    def compute_utility(self, x, alloc, pay):
        """ Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_agents, num_items]
                a: [num_batches, num_agents, num_items]
                p: [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
        """
        return torch.sum(alloc * x, dim=-1) - pay


    def get_misreports(self, x, adv_var, adv_shape):
        """Generate misreports from adversarial variables"""
        num_misreports = adv_shape[1]
        adv = adv_var.unsqueeze(0).repeat(self.config.num_agents, 1, 1, 1, 1)
        x_mis = x.repeat(self.config.num_agents * num_misreports, 1, 1)
        x_r = x_mis.view(adv_shape)
        y = x_r * (1 - self.adv_mask_tensor) + adv * self.adv_mask_tensor
        misreports = y.view(-1, self.config.num_agents, self.config.num_items)
        return x_mis, misreports

    def init_graph(self):
       
        x_shape = [self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_var_shape = [self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]

        # Create masks as tensors
        self.adv_mask = np.zeros(adv_shape)
        self.adv_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0
        self.adv_mask_tensor = torch.tensor(self.adv_mask, dtype=torch.float32)
        
        self.u_mask = np.zeros(u_shape)
        self.u_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0
        self.u_mask_tensor = torch.tensor(self.u_mask, dtype=torch.float32)

        # Adversarial variable (requires grad for optimization)
        self.adv_var = nn.Parameter(torch.zeros(adv_var_shape, dtype=torch.float32))

        if self.mode == "train":

            w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val

            # Lagrange multiplier for regret constraints
            self.w_rgt = nn.Parameter(torch.ones(self.config.num_agents, dtype=torch.float32) * w_rgt_init_val)

            # Update rate (non-trainable parameter)
            self.update_rate = nn.Parameter(torch.tensor(self.config.train.learning_rate, dtype=torch.float32), requires_grad=False)
            self.update_rate_add = self.config.train.up_op_add

            # Weight decay parameter
            wd = None if "wd" not in self.config.train else self.config.train.wd
            
            # Optimizers
            learning_rate = self.config.train.learning_rate
            self.opt_1 = optim.Adam([p for n, p in self.net.named_parameters() if 'w_a' in n or 'w_p' in n or 'b_a' in n or 'b_p' in n], lr=learning_rate, weight_decay=wd if wd else 0)
            self.opt_2 = optim.Adam([self.adv_var], lr=self.config.train.gd_lr)
            self.opt_3 = optim.SGD([self.w_rgt], lr=self.update_rate)

            # Metrics names
            self.metric_names = ["Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "w_rgt_mean", "update_rate"]
        
        elif self.mode == "test":

            # Optimizer for adversarial misreports during test
            self.test_mis_opt = optim.Adam([self.adv_var], lr=self.config.test.gd_lr)

            # Metrics names
            self.metric_names = ["Revenue", "Regret", "IRP"]

    def train(self, generator):
        """
        Runs training
        """
        
        self.train_gen, self.val_gen = generator
        
        iter = self.config.train.restore_iter

        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
            checkpoint = torch.load(model_path)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.w_rgt.data = checkpoint['w_rgt']
            self.update_rate.data = checkpoint['update_rate']
            self.opt_1.load_state_dict(checkpoint['opt_1_state_dict'])
            self.opt_2.load_state_dict(checkpoint['opt_2_state_dict'])
            self.opt_3.load_state_dict(checkpoint['opt_3_state_dict'])

        if iter == 0:
            self.train_gen.save_data(0)
            torch.save({
                'net_state_dict': self.net.state_dict(),
                'w_rgt': self.w_rgt.data,
                'update_rate': self.update_rate.data,
                'opt_1_state_dict': self.opt_1.state_dict(),
                'opt_2_state_dict': self.opt_2.state_dict(),
                'opt_3_state_dict': self.opt_3.state_dict(),
                'iter': iter
            }, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))

        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):
             
            # Get a mini-batch
            X, ADV, perm = next(self.train_gen.gen_func)
            X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=False)
            ADV_tensor = torch.tensor(ADV, dtype=torch.float32)
                
            if iter == 0:
                # Initial Lagrange update
                self.net.eval()
                # Get mechanism for true valuation
                alloc, pay = self.net.inference(X_tensor)
                
                # Get misreports
                adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                
                # Get mechanism for misreports
                a_mis, p_mis = self.net.inference(misreports)
                
                # Utility
                utility = self.compute_utility(X_tensor, alloc, pay)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                
                # Regret Computation
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                
                lag_loss = -torch.sum(self.w_rgt * rgt)
                self.opt_3.zero_grad()
                lag_loss.backward()
                self.opt_3.step()

            self.net.train()
            tic = time.time()    
            
            # Get Best Mis-report
            self.adv_var.data = ADV_tensor
            for _ in range(self.config.train.gd_iter):
                self.opt_2.zero_grad()
                # Compute loss_2: -sum(u_mis) where u_mis is utility from misreports
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, 
                    [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items])
                a_mis, p_mis = self.net.inference(misreports)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                u_mis = (utility_mis.view(u_shape) * self.u_mask_tensor)
                loss_2 = -torch.sum(u_mis)
                loss_2.backward()
                self.opt_2.step()
                self.clip_op_lambda(self.adv_var)
            # Reset optimizer state for adv_var (recreate optimizer)
            self.opt_2 = optim.Adam([self.adv_var], lr=self.config.train.gd_lr)

            if self.config.train.data == "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, self.adv_var.detach().cpu().numpy())

            # Update network params
            self.opt_1.zero_grad()
            # Get mechanism for true valuation
            alloc, pay = self.net.inference(X_tensor)
            
            # Get misreports
            adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
            x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
            
            # Get mechanism for misreports
            a_mis, p_mis = self.net.inference(misreports)
            
            # Utility
            utility = self.compute_utility(X_tensor, alloc, pay)
            utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
            
            # Regret Computation
            u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
            u_mis = (utility_mis.view(u_shape) * self.u_mask_tensor)
            utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
            excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
            rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
            
            # Metrics
            revenue = self.compute_rev(pay)
            rgt_penalty = self.update_rate * torch.sum(rgt ** 2) / 2.0        
            lag_loss = torch.sum(self.w_rgt * rgt)
            loss_1 = -revenue + rgt_penalty + lag_loss
            loss_1.backward()
            self.opt_1.step()
                
            iter += 1

            # Run Lagrange Update
            if iter % self.config.train.update_frequency == 0:
                self.net.eval()
                # Get mechanism for true valuation
                alloc, pay = self.net.inference(X_tensor)
                
                # Get misreports
                adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                
                # Get mechanism for misreports
                a_mis, p_mis = self.net.inference(misreports)
                
                # Utility
                utility = self.compute_utility(X_tensor, alloc, pay)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                
                # Regret Computation
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                # TensorFlow: tf.reduce_max(excess_from_utility, axis=(1, 3))
                # excess_from_utility shape: [num_agents, num_misreports, batch_size, num_agents]
                # After max(dim=1): [num_agents, batch_size, num_agents], removing num_misreports axis
                # After max(dim=2): [num_agents, batch_size], removing last num_agents axis (was axis=3 in original)
                # After mean(dim=1): [num_agents], averaging over batch_size
                rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                
                loss_3 = -torch.sum(self.w_rgt * rgt)
                self.opt_3.zero_grad()
                loss_3.backward()
                self.opt_3.step()

            if iter % self.config.train.up_op_frequency == 0:
                self.update_rate.data += self.update_rate_add

            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter): 
                torch.save({
                    'net_state_dict': self.net.state_dict(),
                    'w_rgt': self.w_rgt.data,
                    'update_rate': self.update_rate.data,
                    'opt_1_state_dict': self.opt_1.state_dict(),
                    'opt_2_state_dict': self.opt_2.state_dict(),
                    'opt_3_state_dict': self.opt_3.state_dict(),
                    'iter': iter
                }, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))
                self.train_gen.save_data(iter)

            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                self.net.eval()
                with torch.no_grad():
                    # Get mechanism for true valuation
                    alloc, pay = self.net.inference(X_tensor)
                    
                    # Get misreports
                    adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                    x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                    
                    # Get mechanism for misreports
                    a_mis, p_mis = self.net.inference(misreports)
                    
                    # Utility
                    utility = self.compute_utility(X_tensor, alloc, pay)
                    utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                    
                    # Regret Computation
                    u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                    utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                    excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                    rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                    
                    # Metrics
                    revenue = self.compute_rev(pay)
                    rgt_mean = torch.mean(rgt)
                    rgt_penalty = self.update_rate * torch.sum(rgt ** 2) / 2.0        
                    lag_loss = torch.sum(self.w_rgt * rgt)
                    loss_1 = -revenue + rgt_penalty + lag_loss
                    
                    metrics = [revenue.item(), rgt_mean.item(), rgt_penalty.item(), lag_loss.item(), loss_1.item(), torch.mean(self.w_rgt).item(), self.update_rate.item()]
                
                fmt_vals = tuple([item for tup in zip(self.metric_names, metrics) for item in tup])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                self.logger.info(log_str)

            if (iter % self.config.val.print_iter) == 0:
                # Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))
                self.net.eval()
                for _ in range(self.config.val.num_batches):
                    X, ADV, _ = next(self.val_gen.gen_func)
                    X_tensor = torch.tensor(X, dtype=torch.float32)
                    ADV_tensor = torch.tensor(ADV, dtype=torch.float32)
                    self.adv_var.data = ADV_tensor
                    val_mis_opt = optim.Adam([self.adv_var], lr=self.config.val.gd_lr)
                    for k in range(self.config.val.gd_iter):
                        val_mis_opt.zero_grad()
                        x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, 
                            [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items])
                        a_mis, p_mis = self.net.inference(misreports)
                        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                        u_mis = (utility_mis.view(u_shape) * self.u_mask_tensor)
                        loss_2 = -torch.sum(u_mis)
                        loss_2.backward()
                        val_mis_opt.step()
                        self.clip_op_lambda(self.adv_var)
                    
                    # Get mechanism for true valuation
                    with torch.no_grad():
                        alloc, pay = self.net.inference(X_tensor)
                        
                        # Get misreports
                        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                        x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                        
                        # Get mechanism for misreports
                        a_mis, p_mis = self.net.inference(misreports)
                        
                        # Utility
                        utility = self.compute_utility(X_tensor, alloc, pay)
                        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                        
                        # Regret Computation
                        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                        utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                        excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                        rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                        
                        # Metrics
                        revenue = self.compute_rev(pay)
                        rgt_mean = torch.mean(rgt)
                        rgt_penalty = self.update_rate * torch.sum(rgt ** 2) / 2.0        
                        lag_loss = torch.sum(self.w_rgt * rgt)
                        loss_1 = -revenue + rgt_penalty + lag_loss
                        
                        metrics = [revenue.item(), rgt_mean.item(), rgt_penalty.item(), lag_loss.item(), loss_1.item(), torch.mean(self.w_rgt).item(), self.update_rate.item()]
                    metric_tot += metrics
                
                metric_tot = metric_tot / self.config.val.num_batches
                fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
                log_str = "VAL-%d" % (iter) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                self.logger.info(log_str)

    def test(self, generator):
        """
        Runs test
        """
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter

        model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
        checkpoint = torch.load(model_path)
        # Load only network parameters (not adv_var, etc.)
        net_state = {k: v for k, v in checkpoint['net_state_dict'].items() if 'w_a' in k or 'w_p' in k or 'b_a' in k or 'b_p' in k}
        self.net.load_state_dict(net_state, strict=False)

        # Test-set Stats
        time_elapsed = 0
            
        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert(hasattr(generator, "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])

        self.net.eval()
        for i in range(self.config.test.num_batches):
            tic = time.time()
            X, ADV, perm = next(self.test_gen.gen_func)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            ADV_tensor = torch.tensor(ADV, dtype=torch.float32)
            self.adv_var.data = ADV_tensor
            test_mis_opt = optim.Adam([self.adv_var], lr=self.config.test.gd_lr)
            for k in range(self.config.test.gd_iter):
                test_mis_opt.zero_grad()
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, 
                    [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items])
                a_mis, p_mis = self.net.inference(misreports)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                u_mis = (utility_mis.view(u_shape) * self.u_mask_tensor)
                loss = -torch.sum(u_mis)
                loss.backward()
                test_mis_opt.step()
                self.clip_op_lambda(self.adv_var)

            # Get mechanism for true valuation
            with torch.no_grad():
                alloc, pay = self.net.inference(X_tensor)
                
                # Get misreports
                adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                
                # Get mechanism for misreports
                a_mis, p_mis = self.net.inference(misreports)
                
                # Utility
                utility = self.compute_utility(X_tensor, alloc, pay)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                
                # Regret Computation
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                # TensorFlow: tf.reduce_max(excess_from_utility, axis=(1, 3))
                # excess_from_utility shape: [num_agents, num_misreports, batch_size, num_agents]
                # After max(dim=1): [num_agents, batch_size, num_agents], removing num_misreports axis
                # After max(dim=2): [num_agents, batch_size], removing last num_agents axis (was axis=3 in original)
                # After mean(dim=1): [num_agents], averaging over batch_size
                rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                
                # Metrics
                revenue = self.compute_rev(pay)
                rgt_mean = torch.mean(rgt)
                irp_mean = torch.mean(F.relu(-utility))
                
                metrics = [revenue.item(), rgt_mean.item(), irp_mean.item()]
                    
                if self.config.test.save_output:
                    alloc_np = alloc.detach().cpu().numpy()
                    pay_np = pay.detach().cpu().numpy()
                    alloc_tst[perm, :, :] = alloc_np
                    pay_tst[perm, :] = pay_np
                        
                metric_tot += metrics
                toc = time.time()
                time_elapsed += (toc - tic)

            fmt_vals = tuple([item for tup in zip(self.metric_names, metrics) for item in tup])
            log_str = "TEST BATCH-%d: t = %.4f" % (i, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
            self.logger.info(log_str)
        
        metric_tot = metric_tot / self.config.test.num_batches
        fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
        log_str = "TEST ALL-%d: t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
        self.logger.info(log_str)
            
        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)
