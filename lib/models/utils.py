import torch
import numpy as np
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # img:    (N, C, H_in, W_in)
    # coords: (N, H_out, W_out, 2)
    # output: (N, C, H_out, W_out)
    
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)


    return img


    