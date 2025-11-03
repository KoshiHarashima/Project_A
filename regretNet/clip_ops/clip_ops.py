from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch


def clip_op_01(x):
    """Clip tensor values to [0.0, 1.0] in-place"""
    x.data.clamp_(min=0.0, max=1.0)
    return x
    
def clip_op_12(x):
    """Clip tensor values to [1.0, 2.0] in-place"""
    x.data.clamp_(min=1.0, max=2.0)
    return x
    
def clip_op_23(x):
    """Clip tensor values to [2.0, 3.0] in-place"""
    x.data.clamp_(min=2.0, max=3.0)
    return x

def clip_op_12_15(x):
    """Clip tensor values: first agent [1.0, 2.0], second agent [1.0, 5.0] in-place"""
    # x shape: [num_misreports, batch_size, num_agents, num_items]
    x[:, :, 0, :].data.clamp_(min=1.0, max=2.0)
    x[:, :, 1, :].data.clamp_(min=1.0, max=5.0)
    return x

def clip_op_416_47(x):
    """Clip tensor values: first item [4.0, 16.0], second item [4.0, 7.0] in-place"""
    # x shape: [num_misreports, batch_size, num_agents, num_items]
    x[:, :, :, 0].data.clamp_(min=4.0, max=16.0)
    x[:, :, :, 1].data.clamp_(min=4.0, max=7.0)
    return x

def clip_op_04_03(x):
    """Clip tensor values: first item [0.0, 4.0], second item [0.0, 3.0] in-place"""
    # x shape: [num_misreports, batch_size, num_agents, num_items]
    x[:, :, :, 0].data.clamp_(min=0.0, max=4.0)
    x[:, :, :, 1].data.clamp_(min=0.0, max=3.0)
    return x

def clip_op_triangle_01_numpy(x):
    """Clip values to triangle region using NumPy (keeps original implementation)"""
    if isinstance(x, torch.Tensor):
        x_np = x.detach().cpu().numpy()
    else:
        x_np = x
    
    x_shape = x_np.shape
    x_flat = np.reshape(x_np, [-1, 2])
    
    invalid_idx = np.where((x_flat[:, 0] < 0) | (x_flat[:, 1] < 0) | (x_flat.sum(-1) >= 1))

    x_invalid = x_flat[invalid_idx]

    if len(x_invalid) > 0:
        p = np.zeros((x_invalid.shape[0], 3, 2))
        d = np.zeros((x_invalid.shape[0], 3))
        t = np.zeros((x_invalid.shape[0], 3))
        t[:, 0] = (x_invalid[:, 0] - x_invalid[:, 1] + 1.0) / 2.0
        t[:, 1] = (1 - x_invalid[:, 1])
        t[:, 2] = (1 - x_invalid[:, 0])
        t = np.clip(t, 0.0, 1.0)

        A = np.array([[0, 1]]).T
        B = np.array([[1, 0]]).T
        O = np.array([[0, 0]]).T
        pts_x = [A, A, B]
        pts_y = [B, O, O]

        for i in range(3):
            p[:, i, :] = ((1 - t[:, i]) * pts_x[i] + t[:, i] * pts_y[i]).T
            d[:, i] = np.sum((x_invalid - p[:, i, :])**2, -1)

        sel_p = p[np.arange(x_invalid.shape[0]), np.argmin(d, -1), :]
        x_flat[invalid_idx] = sel_p
    
    x_np = np.reshape(x_flat, x_shape)
    
    # Convert back to torch tensor if input was torch tensor
    if isinstance(x, torch.Tensor):
        x.data.copy_(torch.tensor(x_np, dtype=x.dtype, device=x.device))
        return x
    else:
        return x_np


def clip_op_triangle_01(x):
    """Clip tensor values to triangle region [0,1]x[0,1] with x1+x2<1"""
    clip_op_triangle_01_numpy(x)
    return x
