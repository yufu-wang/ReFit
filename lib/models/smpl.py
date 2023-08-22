import os
import numpy as np
import torch
from torch.nn import functional as F
import contextlib

from smplx import SMPLLayer as _SMPLLayer
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

from lib.core.constants import JOINT_MAP, JOINT_NAMES


# SMPL data path
SMPL_DATA_PATH = "data/smpl/"

SMPL_MODEL_PATH = os.path.join(SMPL_DATA_PATH, "SMPL_NEUTRAL.pkl")
SMPL_MEAN_PARAMS = os.path.join(SMPL_DATA_PATH, "smpl_mean_params.npz")
SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")
JOINT_REGRESSOR_TRAIN_EXTRA = os.path.join(SMPL_DATA_PATH, 'J_regressor_extra.npy')
JOINT_REGRESSOR_H36M = os.path.join(SMPL_DATA_PATH, 'J_regressor_h36m.npy')

SMPL_MARKER = os.path.join(SMPL_DATA_PATH, 'smpl_ssm.npy')
SMPL_DENSE = os.path.join(SMPL_DATA_PATH, 'smpl_dense.npy')
SMPL_SUB = os.path.join(SMPL_DATA_PATH, 'smpl_down.npy')

# No extra joint regressor for now
class SMPL(_SMPLLayer):

    def __init__(self, *args, **kwargs):
        kwargs["model_path"] = "data/smpl"

        # remove the verbosity for the 10-shapes beta parameters
        with contextlib.redirect_stdout(None):
            super(SMPL, self).__init__(*args, **kwargs)

        # SPIN 49(25 OP + 24) joints
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)
        
        self.ssm = np.load(SMPL_MARKER)
        self.dense = np.load(SMPL_DENSE)
        self.sub = np.load(SMPL_SUB)
        
        
    def forward(self, *args, **kwargs):
        smpl_output = super(SMPL, self).forward(*args, **kwargs)

        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]

        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose,
                            joints=joints)

        return output


    def query(self, hmr_output):
        pred_rotmat = hmr_output['pred_rotmat']
        pred_shape = hmr_output['pred_shape']

        smpl_out = self(global_orient=pred_rotmat[:, [0]],
                        body_pose = pred_rotmat[:, 1:],
                        betas = pred_shape)
        return smpl_out


