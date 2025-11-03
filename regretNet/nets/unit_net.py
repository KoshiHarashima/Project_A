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

        num_a_layers = self.config.net.num_a_layers        
        num_p_layers = self.config.net.num_p_layers

        num_a_hidden_units = self.config.net.num_a_hidden_units
        num_p_hidden_units = self.config.net.num_p_hidden_units
        
        # Alloc network weights and biases
        self.w_a = nn.ParameterList()
        self.b_a = nn.ParameterList()

        # Pay network weights and biases
        self.w_p = nn.ParameterList()
        self.b_p = nn.ParameterList()

        num_in = num_agents * num_items

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
                
        # Output Layer
        w_a_out = nn.Parameter(torch.empty(num_a_hidden_units, (num_agents + 1) * (num_items + 1)))
        if self.init is not None:
            self.init(w_a_out)
        self.w_a.append(w_a_out)

        # Additional parameters for unit-demand networks
        self.wi_a = nn.Parameter(torch.empty(num_a_hidden_units, (num_agents + 1) * (num_items + 1)))
        if self.init is not None:
            self.init(self.wi_a)

        # Biases
        for i in range(num_a_layers - 1):
            b_a_i = nn.Parameter(torch.zeros(num_a_hidden_units))
            self.b_a.append(b_a_i)
                
        b_a_out = nn.Parameter(torch.zeros((num_agents + 1) * (num_items + 1)))
        self.b_a.append(b_a_out)

        self.bi_a = nn.Parameter(torch.zeros((num_agents + 1) * (num_items + 1)))

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

        x_in = x.view(-1, self.config.num_agents * self.config.num_items)

        # Allocation Network
        a = torch.matmul(x_in, self.w_a[0]) + self.b_a[0]
        a = self.activation(a)
        activation_summary(a)
        
        for i in range(1, self.config.net.num_a_layers - 1):
            a = torch.matmul(a, self.w_a[i]) + self.b_a[i]
            a = self.activation(a)
            activation_summary(a)

        agent = torch.matmul(a, self.w_a[-1]) + self.b_a[-1]
        agent = F.softmax(agent.view(-1, self.config.num_agents + 1, self.config.num_items + 1), dim=1)

        item = torch.matmul(a, self.wi_a) + self.bi_a
        item = F.softmax(item.view(-1, self.config.num_agents + 1, self.config.num_items + 1), dim=-1)

        a = torch.minimum(agent, item)
        a = a[:, :self.config.num_agents, :self.config.num_items]  # Equivalent to tf.slice
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
        
        u = torch.sum(a * x.view(-1, self.config.num_agents, self.config.num_items), dim=-1)
        p = p * u
        activation_summary(p)
        
        return a, p
