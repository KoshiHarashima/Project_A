from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
from nets.additive_net import Net as AdditiveNet


class Net(AdditiveNet):
    """
    AdditiveNetを継承し、配分確率の制約違反計算機能を追加
    
    財3の配分確率に対する制約:
    - 下界: max(0, alloc1 + alloc2 - 1) <= alloc3
    - 上界: alloc3 <= min(alloc1, alloc2)
    """
    
    def __init__(self, config):
        super(Net, self).__init__(config)
    
    def compute_allocation_constraint_violation(self, alloc):
        """
        財3の配分確率に対する制約違反を計算
        
        Args:
            alloc: [batch_size, num_agents, num_items]
                   alloc[:, :, 0] = 財1の配分確率
                   alloc[:, :, 1] = 財2の配分確率
                   alloc[:, :, 2] = 財3の配分確率
        
        Returns:
            constraint_violation: [batch_size, num_agents]
                                各サンプル・エージェントごとの制約違反の合計
        """
        alloc1 = alloc[:, :, 0]  # 財1の配分確率
        alloc2 = alloc[:, :, 1]  # 財2の配分確率
        alloc3 = alloc[:, :, 2]  # 財3の配分確率
        
        # 下界: max(0, alloc1 + alloc2 - 1)
        lower_bound = torch.clamp(alloc1 + alloc2 - 1, min=0.0)
        # 上界: min(alloc1, alloc2)
        upper_bound = torch.minimum(alloc1, alloc2)
        
        # 制約違反
        lower_violation = F.relu(lower_bound - alloc3)  # 下界違反（alloc3が小さすぎる）
        upper_violation = F.relu(alloc3 - upper_bound)   # 上界違反（alloc3が大きすぎる）
        
        constraint_violation = lower_violation + upper_violation
        
        return constraint_violation

