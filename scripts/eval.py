import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from os.path import join
from time import time
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from lib.datasets.base_dataset import BaseDataset
from lib.utils.rotation_conversions import axis_angle_to_matrix
from lib.utils.pose_utils import Evaluator

from lib.core import constants
from lib.core.config import parse_args
from lib import get_model


# Dataloaders
db = BaseDataset('3dpw_test', is_train=False, use_augmentation=False, normalization=True, cropped=False, crop_size=256)
data_loader = DataLoader(db, batch_size=32, shuffle=False, num_workers=12)


# Configuration
cfg_args = ['--cfg', 'configs/config.yaml']
cfg = parse_args(cfg_args)
cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model
model = get_model(cfg)
checkpoint = 'data/pretrain/refit_all/checkpoint_best.pth.tar'
state_dict = torch.load(checkpoint, map_location=cfg.DEVICE)
_ = model.load_state_dict(state_dict['model'], strict=False)
print("Loaded checkpoint", checkpoint)

model = model.to(cfg.DEVICE)
model.eval()


# Run model and evaluation
evaluator = Evaluator(dataset_length=len(db))
J_regressor = db.J_regressor.to(cfg.DEVICE)
num_iter = 5

print('Evaluation best model of config default')
print('Inference iteration number: {}'.format(num_iter))
for batch in tqdm(data_loader):
    batch = {k: v.to(cfg.DEVICE) for k, v in batch.items() if type(v)==torch.Tensor}

    # gt joints
    gt_keypoints_3d = batch['pose_3d']
    gt_verts = batch['gt_verts']

    # prediction
    with torch.no_grad():
        out, _ = model(batch, iters=num_iter, flip_test=True)
        
        smpl_out = model.smpl.query(out)
        pred_vertices = smpl_out.vertices
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_vertices = pred_vertices - pred_pelvis

    # evaluation
    evaluator(gt_keypoints_3d, pred_keypoints_3d, '3dpw', gt_verts, pred_vertices)


print('Results for model')
evaluator.log()
print()




