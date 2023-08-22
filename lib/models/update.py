import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_linear import MultiLinear


class GRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=128):
        super(GRU, self).__init__()
        self.lz = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.lr = nn.Linear(hidden_dim+input_dim, hidden_dim)
        self.lq = nn.Linear(hidden_dim+input_dim, hidden_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)

        z = torch.sigmoid(self.lz(hx))
        r = torch.sigmoid(self.lr(hx))
        q = torch.tanh(self.lq(torch.cat([r*h, x], dim=-1)))

        h = (1-z) * h + z * q
        return h


class MultiGRU(nn.Module):
    def __init__(self, n_head=24, hidden_dim=12, input_dim=256):
        super(MultiGRU, self).__init__()
        self.lz = MultiLinear(n_head, hidden_dim+input_dim, hidden_dim)
        self.lr = MultiLinear(n_head, hidden_dim+input_dim, hidden_dim)
        self.lq = MultiLinear(n_head, hidden_dim+input_dim, hidden_dim)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=-1)

        z = torch.sigmoid(self.lz(hx))
        r = torch.sigmoid(self.lr(hx))
        q = torch.tanh(self.lq(torch.cat([r*h, x], dim=-1)))

        h = (1-z) * h + z * q
        return h


class UpdateBlock(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=32):
        super(UpdateBlock, self).__init__()

        self.pose_gru = MultiGRU(24, hidden_dim, input_dim)
        self.shape_gru = GRU(hidden_dim, input_dim)
        self.cam_gru = GRU(hidden_dim, input_dim)

    def forward(self, hpose, hshape, hcam, 
                    loc, pose, shape, cam):

        x = torch.cat([loc, pose, shape, cam], dim=-1)

        hpose = self.pose_gru(hpose, x[:,None,:].repeat(1,24,1))
        hshape = self.shape_gru(hshape, x)
        hcam = self.cam_gru(hcam, x)

        return hpose, hshape, hcam


class Regressor(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=32, num_layer=1, pose_dim=6):
        super(Regressor, self).__init__()
        input_dim = input_dim + 3

        self.p = self._make_multilinear(num_layer, 24, input_dim, hidden_dim)
        self.s = self._make_linear(num_layer, input_dim, hidden_dim)
        self.c = self._make_linear(num_layer, input_dim, hidden_dim)

        self.decpose = MultiLinear(24, hidden_dim, pose_dim)
        self.decshape = nn.Linear(hidden_dim, 10)
        self.deccam = nn.Linear(hidden_dim, 3)

    def forward(self, hpose, hshape, hcam, bbox_info):
        BN = hpose.shape[0]

        hpose = torch.cat([hpose, bbox_info.unsqueeze(1).repeat(1,24,1)], -1)
        hshape = torch.cat([hshape, bbox_info], -1)
        hcam = torch.cat([hcam, bbox_info], -1)
        
        d_pose = self.decpose(self.p(hpose)).view(BN, -1)
        d_shape = self.decshape(self.s(hshape))
        d_cam = self.deccam(self.c(hcam))

        return d_pose, d_shape, d_cam
    
    def _make_linear(self, num, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [nn.Linear(plane, hidden_dim), 
                     nn.ReLU(inplace=True)]
            layers.extend(layer)  

            plane = hidden_dim

        return nn.Sequential(*layers)
    
    def _make_multilinear(self, num, n_head, input_dim, hidden_dim):
        plane = input_dim
        layers = []
        for i in range(num):
            layer = [MultiLinear(n_head, plane, hidden_dim), 
                     nn.ReLU(inplace=True)]
            layers.extend(layer)
            
            plane = hidden_dim

        return nn.Sequential(*layers) 


