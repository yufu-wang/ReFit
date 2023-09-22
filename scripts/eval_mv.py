import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from lib import get_model
from lib.core.config import parse_args
from lib.datasets.mv_dataset import MVDataset, MV_Sampler
from lib.utils.pose_utils import Evaluator
from lib.utils.geometry import avg_rot

parser = argparse.ArgumentParser()
parser.add_argument('--pred_avg', action='store_true', help='prediction averaging across multiviews')
parser.add_argument('--mv_refit', action='store_true', help='multi-view refit')
parser.add_argument('--num_iter', type=int, default=5, help='number of iterations')
args = parser.parse_args()


# Configuration
cfg_args = ['--cfg', 'configs/config.yaml']
cfg = parse_args(cfg_args)
cfg.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.mv_refit:
    cfg.MODEL.VERSION = 'mv'


# Dataloaders
mv_db = MVDataset('h36m-mv', is_train=False, use_augmentation=False, 
                    normalization=True, cropped=False, crop_size=256)

db = mv_db.dataset
mv_idx = mv_db.mv_idx

np.random.seed(0)
subset = np.random.permutation(len(mv_idx))[:2000]
mv_subset = mv_idx[subset]

mv_sampler = MV_Sampler(mv_subset)
data_loader = DataLoader(db, batch_size=4, sampler=mv_sampler, shuffle=False, num_workers=10)


# Model
model = get_model(cfg)
checkpoint = 'results/refit_h36m/checkpoint_best.pth.tar'
state_dict = torch.load(checkpoint, map_location=cfg.DEVICE)
_ = model.load_state_dict(state_dict['model'], strict=False)
print("Loaded checkpoint", checkpoint)

model = model.to(cfg.DEVICE)
model.eval()


# Run model and evaluation
evaluator = Evaluator(dataset_length=len(db))
J_regressor = db.J_regressor.to(cfg.DEVICE)
num_iter = args.num_iter

for batch in tqdm(data_loader):
    batch = {k: v.to(cfg.DEVICE) for k, v in batch.items() if type(v)==torch.Tensor}

    # gt joints
    gt_keypoints_3d = batch['pose_3d']
    gt_verts = batch['gt_verts']

    # prediction
    with torch.no_grad():
        out, _ = model(batch, iters=num_iter)

        if args.pred_avg:
            # average rotation
            pred_rotmat = out['pred_rotmat']
            bn = gt_keypoints_3d.size(0)

            cam_R = batch['cam_R']
            global_rotmat = pred_rotmat[:, 0]
            global_rotmat_w = torch.einsum('bij, bjk -> bik', cam_R.transpose(-1,-2), global_rotmat)
            global_rotmat_w = avg_rot(global_rotmat_w)
            global_rotmat = torch.einsum('bij, jk -> bik', cam_R, global_rotmat_w)
            global_rotmat = global_rotmat[:, np.newaxis]

            body_rotmat = pred_rotmat[:, 1:]
            body_rotmat = avg_rot(body_rotmat)
            body_rotmat = body_rotmat[np.newaxis].repeat(bn,1,1,1)
            avg_rotmat = torch.concat([global_rotmat, body_rotmat], dim=1)

            # average shape
            pred_shape = out['pred_shape']
            avg_shape = pred_shape.mean(dim=0, keepdim=True).repeat(bn,1)

            out = {}
            out['pred_rotmat'] = avg_rotmat
            out['pred_shape'] = avg_shape

        
        smpl_out = model.smpl.query(out)
        pred_vertices = smpl_out.vertices
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1)

        pred_keypoints_3d = torch.matmul(J_regressor_batch, pred_vertices)
        pred_pelvis = pred_keypoints_3d[:, [0],:].clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
        pred_vertices = pred_vertices - pred_pelvis

    # evaluation
    evaluator(gt_keypoints_3d, pred_keypoints_3d, '3dpw', gt_verts, pred_vertices)


print('Results for model H36M')
evaluator.log()
print()




