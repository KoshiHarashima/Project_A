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
from trainer.trainer import Trainer as BaseTrainer


class Trainer(BaseTrainer):
    """
    BaseTrainerを継承し、配分確率制約違反の罰則項を追加
    
    財3の配分確率に対する制約:
    - 下界: max(0, alloc1 + alloc2 - 1) <= alloc3
    - 上界: alloc3 <= min(alloc1, alloc2)
    """
    
    def __init__(self, config, mode, net, clip_op_lambda):
        super(Trainer, self).__init__(config, mode, net, clip_op_lambda)
    
    def compute_allocation_constraint_violation(self, alloc):
        """
        ネットワークの制約違反計算関数を呼び出し
        """
        if hasattr(self.net, 'compute_allocation_constraint_violation'):
            return self.net.compute_allocation_constraint_violation(alloc)
        else:
            # フォールバック（通常は発生しない）
            return torch.zeros(alloc.shape[0], alloc.shape[1])
    
    def init_graph(self):
        """
        BaseTrainerのinit_graphを呼び出し、制約違反用のパラメータを追加
        """
        super(Trainer, self).init_graph()
        
        if self.mode == "train":
            # 制約違反用のLagrange乗数
            w_constraint_init_val = 5.0 if "w_constraint_init_val" not in self.config.train else self.config.train.w_constraint_init_val
            self.w_constraint = nn.Parameter(
                torch.ones(self.config.num_agents, dtype=torch.float32, device=self.device) * w_constraint_init_val
            )
            
            # 制約違反用の更新レート
            constraint_update_rate = 1.0 if "constraint_update_rate" not in self.config.train else self.config.train.constraint_update_rate
            self.constraint_update_rate = nn.Parameter(
                torch.tensor(constraint_update_rate, dtype=torch.float32, device=self.device),
                requires_grad=False
            )
            
            # 制約違反用のオプティマイザー
            self.opt_constraint = optim.SGD([self.w_constraint], lr=self.constraint_update_rate.item())
            
            # メトリクス名に追加
            self.metric_names = [
                "Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss",
                "w_rgt_mean", "update_rate",
                "Constraint_Viol", "Const_Penalty", "Const_Lag_Loss", "w_constraint_mean"
            ]
    
    def train(self, generator):
        """
        BaseTrainerのtrainをオーバーライドし、制約違反の処理を追加
        """
        self.train_gen, self.val_gen = generator
        
        iter = self.config.train.restore_iter
        
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt')
            checkpoint = torch.load(model_path, map_location=self.device)
            self.net.load_state_dict(checkpoint['net_state_dict'])
            self.w_rgt.data = checkpoint['w_rgt'].to(self.device)
            self.update_rate.data = checkpoint['update_rate'].to(self.device)
            if 'w_constraint' in checkpoint:
                self.w_constraint.data = checkpoint['w_constraint'].to(self.device)
            if 'constraint_update_rate' in checkpoint:
                self.constraint_update_rate.data = checkpoint['constraint_update_rate'].to(self.device)
            self.opt_1.load_state_dict(checkpoint['opt_1_state_dict'])
            self.opt_2.load_state_dict(checkpoint['opt_2_state_dict'])
            self.opt_3.load_state_dict(checkpoint['opt_3_state_dict'])
            if 'opt_constraint_state_dict' in checkpoint:
                self.opt_constraint.load_state_dict(checkpoint['opt_constraint_state_dict'])
        
        if iter == 0:
            self.train_gen.save_data(0)
            save_dict = {
                'net_state_dict': self.net.state_dict(),
                'w_rgt': self.w_rgt.data,
                'update_rate': self.update_rate.data,
                'w_constraint': self.w_constraint.data,
                'constraint_update_rate': self.constraint_update_rate.data,
                'opt_1_state_dict': self.opt_1.state_dict(),
                'opt_2_state_dict': self.opt_2.state_dict(),
                'opt_3_state_dict': self.opt_3.state_dict(),
                'opt_constraint_state_dict': self.opt_constraint.state_dict(),
                'iter': iter
            }
            torch.save(save_dict, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))
        
        time_elapsed = 0.0
        while iter < (self.config.train.max_iter):
            
            # Get a mini-batch
            X, ADV, perm = next(self.train_gen.gen_func)
            X_tensor = torch.from_numpy(X).to(self.device).float()
            ADV_tensor = torch.from_numpy(ADV).to(self.device).float()
                
            if iter == 0:
                # Initial Lagrange update (regret and constraint)
                self.net.eval()
                alloc, pay = self.net.inference(X_tensor)
                
                adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                a_mis, p_mis = self.net.inference(misreports)
                
                utility = self.compute_utility(X_tensor, alloc, pay)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                
                lag_loss = -torch.sum(self.w_rgt * rgt)
                self.opt_3.zero_grad()
                lag_loss.backward()
                self.opt_3.step()
                
                # Constraint violation initial update
                constraint_violation = self.compute_allocation_constraint_violation(alloc)
                constraint_violation_mean = torch.mean(constraint_violation, dim=0)
                constraint_lag_loss = -torch.sum(self.w_constraint * constraint_violation_mean)
                self.opt_constraint.zero_grad()
                constraint_lag_loss.backward()
                self.opt_constraint.step()
            
            self.net.train()
            tic = time.time()
            
            # Get Best Mis-report
            self.adv_var.data.copy_(ADV_tensor)
            for _ in range(self.config.train.gd_iter):
                self.opt_2.zero_grad()
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
            self.opt_2 = optim.Adam([self.adv_var], lr=self.config.train.gd_lr)
            
            if self.config.train.data == "fixed" and self.config.train.adv_reuse:
                # CPU only: .cpu() is unnecessary but harmless (no-op)
                self.train_gen.update_adv(perm, self.adv_var.detach().numpy())
            
            # Update network params
            self.opt_1.zero_grad()
            alloc, pay = self.net.inference(X_tensor)
            
            adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
            x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
            a_mis, p_mis = self.net.inference(misreports)
            
            utility = self.compute_utility(X_tensor, alloc, pay)
            utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
            
            u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
            u_mis = (utility_mis.view(u_shape) * self.u_mask_tensor)
            utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
            excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
            rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
            
            # Constraint violation computation
            constraint_violation = self.compute_allocation_constraint_violation(alloc)
            constraint_violation_mean = torch.mean(constraint_violation, dim=0)
            
            # Metrics
            revenue = self.compute_rev(pay)
            rgt_penalty = self.update_rate * torch.sum(rgt ** 2) / 2.0
            lag_loss = torch.sum(self.w_rgt * rgt)
            constraint_penalty = self.constraint_update_rate * torch.sum(constraint_violation_mean ** 2) / 2.0
            constraint_lag_loss = torch.sum(self.w_constraint * constraint_violation_mean)
            loss_1 = -revenue + rgt_penalty + lag_loss + constraint_penalty + constraint_lag_loss
            loss_1.backward()
            self.opt_1.step()
            
            iter += 1
            
            # Run Lagrange Update
            if iter % self.config.train.update_frequency == 0:
                self.net.eval()
                alloc, pay = self.net.inference(X_tensor)
                
                adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                a_mis, p_mis = self.net.inference(misreports)
                
                utility = self.compute_utility(X_tensor, alloc, pay)
                utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                
                u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                
                loss_3 = -torch.sum(self.w_rgt * rgt)
                self.opt_3.zero_grad()
                loss_3.backward()
                self.opt_3.step()
                
                # Constraint violation update
                constraint_violation = self.compute_allocation_constraint_violation(alloc)
                constraint_violation_mean = torch.mean(constraint_violation, dim=0)
                loss_constraint = -torch.sum(self.w_constraint * constraint_violation_mean)
                self.opt_constraint.zero_grad()
                loss_constraint.backward()
                self.opt_constraint.step()
            
            if iter % self.config.train.up_op_frequency == 0:
                self.update_rate.data += self.update_rate_add
                # Update optimizer learning rates
                for param_group in self.opt_3.param_groups:
                    param_group['lr'] = self.update_rate.item()
                for param_group in self.opt_constraint.param_groups:
                    param_group['lr'] = self.constraint_update_rate.item()
            
            toc = time.time()
            time_elapsed += (toc - tic)
                        
            if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter):
                save_dict = {
                    'net_state_dict': self.net.state_dict(),
                    'w_rgt': self.w_rgt.data,
                    'update_rate': self.update_rate.data,
                    'w_constraint': self.w_constraint.data,
                    'constraint_update_rate': self.constraint_update_rate.data,
                    'opt_1_state_dict': self.opt_1.state_dict(),
                    'opt_2_state_dict': self.opt_2.state_dict(),
                    'opt_3_state_dict': self.opt_3.state_dict(),
                    'opt_constraint_state_dict': self.opt_constraint.state_dict(),
                    'iter': iter
                }
                torch.save(save_dict, os.path.join(self.config.dir_name, 'model-' + str(iter) + '.pt'))
                self.train_gen.save_data(iter)
            
            if (iter % self.config.train.print_iter) == 0:
                # Train Set Stats
                self.net.eval()
                with torch.no_grad():
                    alloc, pay = self.net.inference(X_tensor)
                    
                    adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                    x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                    a_mis, p_mis = self.net.inference(misreports)
                    
                    utility = self.compute_utility(X_tensor, alloc, pay)
                    utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                    
                    u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                    utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                    excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                    rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                    
                    constraint_violation = self.compute_allocation_constraint_violation(alloc)
                    constraint_violation_mean = torch.mean(constraint_violation, dim=0)
                    
                    revenue = self.compute_rev(pay)
                    rgt_mean = torch.mean(rgt)
                    rgt_penalty = self.update_rate * torch.sum(rgt ** 2) / 2.0
                    lag_loss = torch.sum(self.w_rgt * rgt)
                    constraint_penalty = self.constraint_update_rate * torch.sum(constraint_violation_mean ** 2) / 2.0
                    constraint_lag_loss = torch.sum(self.w_constraint * constraint_violation_mean)
                    loss_1 = -revenue + rgt_penalty + lag_loss + constraint_penalty + constraint_lag_loss
                    
                    metrics = [
                        revenue.item(), rgt_mean.item(), rgt_penalty.item(), lag_loss.item(), loss_1.item(),
                        torch.mean(self.w_rgt).item(), self.update_rate.item(),
                        constraint_violation_mean.item(), constraint_penalty.item(), constraint_lag_loss.item(),
                        torch.mean(self.w_constraint).item()
                    ]
                
                fmt_vals = tuple([item for tup in zip(self.metric_names, metrics) for item in tup])
                log_str = "TRAIN-BATCH Iter: %d, t = %.4f" % (iter, time_elapsed) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                self.logger.info(log_str)
            
            if (iter % self.config.val.print_iter) == 0:
                # Validation Set Stats
                metric_tot = np.zeros(len(self.metric_names))
                self.net.eval()
                for _ in range(self.config.val.num_batches):
                    X, ADV, _ = next(self.val_gen.gen_func)
                    X_tensor = torch.from_numpy(X).to(self.device).float()
                    ADV_tensor = torch.from_numpy(ADV).to(self.device).float()
                    self.adv_var.data.copy_(ADV_tensor)
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
                    
                    with torch.no_grad():
                        alloc, pay = self.net.inference(X_tensor)
                        
                        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
                        x_mis, misreports = self.get_misreports(X_tensor, self.adv_var, adv_shape)
                        a_mis, p_mis = self.net.inference(misreports)
                        
                        utility = self.compute_utility(X_tensor, alloc, pay)
                        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)
                        
                        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]
                        utility_true = utility.repeat(self.config.num_agents * self.config[self.mode].num_misreports, 1)
                        excess_from_utility = F.relu((utility_mis - utility_true).view(u_shape) * self.u_mask_tensor)
                        rgt = torch.mean(torch.max(excess_from_utility, dim=1)[0].max(dim=2)[0], dim=1)
                        
                        constraint_violation = self.compute_allocation_constraint_violation(alloc)
                        constraint_violation_mean = torch.mean(constraint_violation, dim=0)
                        
                        revenue = self.compute_rev(pay)
                        rgt_mean = torch.mean(rgt)
                        rgt_penalty = self.update_rate * torch.sum(rgt ** 2) / 2.0
                        lag_loss = torch.sum(self.w_rgt * rgt)
                        constraint_penalty = self.constraint_update_rate * torch.sum(constraint_violation_mean ** 2) / 2.0
                        constraint_lag_loss = torch.sum(self.w_constraint * constraint_violation_mean)
                        loss_1 = -revenue + rgt_penalty + lag_loss + constraint_penalty + constraint_lag_loss
                        
                        metrics = [
                            revenue.item(), rgt_mean.item(), rgt_penalty.item(), lag_loss.item(), loss_1.item(),
                            torch.mean(self.w_rgt).item(), self.update_rate.item(),
                            constraint_violation_mean.item(), constraint_penalty.item(), constraint_lag_loss.item(),
                            torch.mean(self.w_constraint).item()
                        ]
                    metric_tot += metrics
                
                metric_tot = metric_tot / self.config.val.num_batches
                fmt_vals = tuple([item for tup in zip(self.metric_names, metric_tot) for item in tup])
                log_str = "VAL-%d" % (iter) + ", %s: %.6f" * len(self.metric_names) % fmt_vals
                self.logger.info(log_str)

