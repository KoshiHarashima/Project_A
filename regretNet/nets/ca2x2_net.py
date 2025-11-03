from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import *


class Net(BaseNet):

    def __init__(self, config):
        super(Net, self).__init__(config)
        self.build_net()

    def build_net(self):
        """
        Initializes network variables
        """

        num_agents = self.config.num_agents
        num_items = self.config.num_items

        num_a_hidden_units = self.config.net.num_a_hidden_units        
        num_p_hidden_units = self.config.net.num_p_hidden_units

        num_p_layers = self.config.net.num_p_layers
        num_a_layers = self.config.net.num_a_layers

        assert(num_agents == 2), "Only supports num_agents = 2"
        assert(num_items == 2), "Only supports num_items = 2"
        
        # Alloc network weights and biases
        self.w_a = nn.ParameterList()
        self.b_a = nn.ParameterList()

        # Pay network weights and biases
        self.w_p = nn.ParameterList()
        self.b_p = nn.ParameterList()

        num_in = 6

        # Allocation network layers
        # Input Layer
        w_a_0 = nn.Parameter(torch.empty(num_in, num_a_hidden_units))
        if self.init is not None:
            self.init(w_a_0)
        self.w_a.append(w_a_0)

        # Hidden Layers
        for i in range(1, num_a_layers - 1):
            w_a_i = nn.Parameter(torch.empty(num_a_hidden_units, num_a_hidden_units))
            if self.init is not None:
                self.init(w_a_i)
            self.w_a.append(w_a_i)

        # Last Layer alloc weights (CA-specific structure)
        self.wi1_a = nn.Parameter(torch.empty(num_a_hidden_units, 5))
        if self.init is not None:
            self.init(self.wi1_a)

        self.wi2_a = nn.Parameter(torch.empty(num_a_hidden_units, 5))
        if self.init is not None:
            self.init(self.wi2_a)

        self.wa1_a = nn.Parameter(torch.empty(num_a_hidden_units, 3))
        if self.init is not None:
            self.init(self.wa1_a)

        self.wa2_a = nn.Parameter(torch.empty(num_a_hidden_units, 3))
        if self.init is not None:
            self.init(self.wa2_a)

        # Biases
        for i in range(num_a_layers - 1):
            b_a_i = nn.Parameter(torch.zeros(num_a_hidden_units))
            self.b_a.append(b_a_i)

        # Last Layer alloc bias
        self.bi1_a = nn.Parameter(torch.zeros(5))
        self.bi2_a = nn.Parameter(torch.zeros(5))
        self.ba1_a = nn.Parameter(torch.zeros(3))
        self.ba2_a = nn.Parameter(torch.zeros(3))

        # Payment network layers
        # Input Layer
        w_p_0 = nn.Parameter(torch.empty(num_in, num_p_hidden_units))
        if self.init is not None:
            self.init(w_p_0)
        self.w_p.append(w_p_0)

        # Hidden Layers
        for i in range(1, num_p_layers - 1):
            w_p_i = nn.Parameter(torch.empty(num_p_hidden_units, num_p_hidden_units))
            if self.init is not None:
                self.init(w_p_i)
            self.w_p.append(w_p_i)
                
        # Output Layer
        w_p_out = nn.Parameter(torch.empty(num_p_hidden_units, num_agents))
        if self.init is not None:
            self.init(w_p_out)
        self.w_p.append(w_p_out)

        # Biases
        for i in range(num_p_layers - 1):
            b_p_i = nn.Parameter(torch.zeros(num_p_hidden_units))
            self.b_p.append(b_p_i)
                
        b_p_out = nn.Parameter(torch.zeros(num_agents))
        self.b_p.append(b_p_out)

    def inference(self, x):
        """
        Inference
        """
        # Ensure x is a torch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x_in = x.view(-1, 6)

        # Allocation Network
        a = torch.matmul(x_in, self.w_a[0]) + self.b_a[0]
        a = self.activation(a)
        activation_summary(a)
        
        for i in range(1, self.config.net.num_a_layers - 1):
            a = torch.matmul(a, self.w_a[i]) + self.b_a[i]
            a = self.activation(a)
            activation_summary(a)

        # From Zhe's code - CA-specific allocation computation
        a_item1_ = F.softmax(torch.matmul(a, self.wi1_a) + self.bi1_a, dim=-1)
        a_item2_ = F.softmax(torch.matmul(a, self.wi2_a) + self.bi2_a, dim=-1)
        
        a_agent1_bundle = F.softmax(torch.matmul(a, self.wa1_a) + self.ba1_a, dim=-1)
        a_agent2_bundle = F.softmax(torch.matmul(a, self.wa2_a) + self.ba2_a, dim=-1)
        
        # Concatenate slices: [item1_col0, item2_col0, min(item1_col2, item2_col2)]
        a_agent1_ = torch.cat([
            a_item1_[:, 0:1],  # slice [0,0] -> [:, 0:1]
            a_item2_[:, 0:1],  # slice [0,0] -> [:, 0:1]
            torch.minimum(a_item1_[:, 2:3], a_item2_[:, 2:3])  # slice [0,2] -> [:, 2:3]
        ], dim=1)
        
        # Concatenate slices: [item1_col1, item2_col1, min(item1_col3, item2_col3)]
        a_agent2_ = torch.cat([
            a_item1_[:, 1:2],  # slice [0,1] -> [:, 1:2]
            a_item2_[:, 1:2],  # slice [0,1] -> [:, 1:2]
            torch.minimum(a_item1_[:, 3:4], a_item2_[:, 3:4])  # slice [0,3] -> [:, 3:4]
        ], dim=1)
                
        a_agent1 = torch.minimum(a_agent1_, a_agent1_bundle)
        a_agent2 = torch.minimum(a_agent2_, a_agent2_bundle)
        
        a = torch.cat([a_agent1, a_agent2], dim=1).view(-1, 2, 3)
        # Zhe's code End

        activation_summary(a)

        # Payment Network
        p = torch.matmul(x_in, self.w_p[0]) + self.b_p[0]
        p = self.activation(p)
        activation_summary(p)

        for i in range(1, self.config.net.num_p_layers - 1):
            p = torch.matmul(p, self.w_p[i]) + self.b_p[i]
            p = self.activation(p)
            activation_summary(p)

        p = torch.matmul(p, self.w_p[-1]) + self.b_p[-1]
        p = torch.sigmoid(p)
        activation_summary(p)
        
        u = torch.sum(a * x.view(-1, 2, 3), dim=-1)
        p = p * u
        activation_summary(p)
        
        return a, p
