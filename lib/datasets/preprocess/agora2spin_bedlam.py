"""
Export AGORA to SPIN style npz file
"""

import numpy as np
import torch
import os
import pickle
from os.path import join
import tqdm
from tqdm import tqdm

from lib.utils import rotation_conversions as geo
from lib.models.smpl import SMPL
from .agora import load_agora_smpl
from .agora_projection import *


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1]).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return R, t, S1_hat.permute(0, 2, 1)


def convert_agora():
    for idx in tqdm(range(len(df))):
        for pNum in range(len(df.iloc[idx]['isValid'])):

            sample = df.iloc[idx]
            imgname = join(f'train_{train_idx}', 
                           sample.imgPath.replace('.png', '_1280x720.png'))
        
            if (sample.isValid[pNum]):
                annfile = sample.gt_path_smpl[pNum]
                annpath = join(root, annfile)
                ann = load_agora_smpl(annpath)

                is_kid = sample.kid[pNum]

                # No kid for Bedlam training
                if is_kid:
                    continue

                gt_smpl = {'pred_rotmat': ann['full_pose'].detach(), 'pred_shape': ann['betas'][:,:10].detach()}
                smpl_out = smpl.query(gt_smpl)
                verts = smpl_out.vertices + ann['translation'].detach()
                joints = smpl_out.joints + ann['translation'].detach()

                # Projection
                joints2d_cam, joints3d_cam, focal = project_2d(df, idx, pNum, joints[0]) 

                # Kid Projection
                if is_kid:
                    kid_smpl = {'pred_rotmat': ann['full_pose'].detach(), 'pred_shape': ann['betas'].detach()}
                    kid_out = smpl_kid.query(kid_smpl)
                    kid_joints = kid_out.joints + ann['translation'].detach()
                    joints2d_cam, joints3d_cam, _ = project_2d(df, idx, pNum, kid_joints[0]) 
                
                # GT joints2d
                joints2d = joints2d_cam[25:]
                joints2d = np.concatenate([joints2d, np.ones([24, 1])], axis=1)
                
                ######## Some processing ########
                x = joints2d[:, 0]
                y = joints2d[:, 1]
                valid = (0<=x) * (x<=1280) * (0<=y) * (y<=720) 
                joints2d[~valid] *= 0
                if valid.sum() <= 0:
                    continue
                ########################################
                
                
                # GT joints3d (I think it's okay not to align it as well)
                joints3d = joints3d_cam[25:]
                pelvis = joints3d[[14]]
                
                joints3d = joints3d - pelvis
                joints3d = joints3d.detach().numpy()
                joints3d = np.concatenate([joints3d, np.ones([24, 1])], axis=1)
                
                
                # Center and scale
                part = joints2d[valid]
                bbox = [min(part[:,0]), min(part[:,1]),
                        max(part[:,0]), max(part[:,1])]
                
                center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200
                if scale <= 0.3:
                    continue
                
                # SMPL pose and shape
                ######## Some processing ########
                joints_root = joints[0][25:] - joints[0][[8]]
                joints3d_root = joints3d_cam[25:] - joints3d_cam[[8]]

                joints_root = joints_root.float()
                joints3d_root = joints3d_root.float()
                R, _, _ = compute_similarity_transform(joints_root[None], joints3d_root[None])

                root_pose = ann['root_pose'][0]
                root_rotmat = geo.axis_angle_to_matrix(root_pose)
                root_rotmat = R @ root_rotmat
                root_pose = geo.matrix_to_axis_angle(root_rotmat)
                ########################################
                
                pose = torch.concat([root_pose, ann['body_pose'][0]])
                pose = pose.reshape(-1).detach().numpy()
                shape = ann['betas'][0, :10].detach().numpy()
                
                
                ######### Get Default SMPL 24 joints #########

                # if is_kid:
                #     kid_smpl = {'pred_rotmat': ann['full_pose'].detach(), 'pred_shape': ann['betas'].detach()}
                #     kid_out = smpl_kid.query_default(kid_smpl)
                #     kid_joints = kid_out.joints + ann['translation'].detach()
                #     joints2d_cam, joints3d_cam, _ = project_2d(df, idx, pNum, kid_joints[0]) 
                # else:
                #     gt_smpl = {'pred_rotmat': ann['full_pose'].detach(), 'pred_shape': ann['betas'][:,:10].detach()}
                #     smpl_out = smpl.query_default(gt_smpl)
                #     joints = smpl_out.joints + ann['translation'].detach()
                #     joints2d_cam, joints3d_cam, _ = project_2d(df, idx, pNum, joints[0]) 

                
                # # GT joints2d
                # joints2d = joints2d_cam[:24]
                # joints2d = np.concatenate([joints2d, np.ones([24, 1])], axis=1)

                
                # # GT joints3d
                # joints3d = joints3d_cam[:24]
                # pelvis = joints3d[[0]]
                # joints3d = joints3d - pelvis
                # joints3d = joints3d.detach().numpy()
                # joints3d = np.concatenate([joints3d, np.ones([24, 1])], axis=1)

                ########################################################


                # All
                imgnames_.append(imgname)
                centers_.append(center)
                scales_.append(scale)
                S_.append(joints3d)
                parts_.append(joints2d)
                
                poses_.append(pose)
                shapes_.append(shape)
                img_focals_.append(focal)
                img_centers_.append(img_center)
                is_kid_.append(is_kid)

                if is_kid:
                    kid_shapes_.append(ann['betas'][0].detach().numpy())
                else:
                    kid_shapes_.append(np.zeros([11]))



if __name__ == '__main__':
    # ROOT
    root = '/Users/yufu/vision_database/agora'
    smpl = SMPL()
    smpl_kid = SMPL(model_type='smpl', 
                    gender='neutral',
                    age='kid', 
                    kid_template_path='data/smpl/smpl_kid_template.npy',
                    ext='npz')

    # scale factor
    scaleFactor = 1.2
    img_center = [1280/2., 720/2.]

    # structs we use
    imgnames_, scales_, centers_, S_, parts_ = [], [], [], [], []
    poses_, shapes_ = [], []
    img_focals_, img_centers_ = [], []
    is_kid_, kid_shapes_ = [], []

    for train_idx in range(10):

        # DF file
        file = f'data/agora/Cam/train_{train_idx}.pkl'
        with open(file, 'rb') as f:
            df = pickle.load(f)

        print(f"Converting train_{train_idx}...")
        convert_agora()

    out_file = 'data/dataset_extras/agora_train.npz'
    np.savez(out_file, imgname = imgnames_,
                       center = centers_,
                       scale = scales_,
                       S = S_,
                       part = parts_,
                       pose = poses_,
                       shape = shapes_,
                       img_focal = img_focals_,
                       img_center = img_centers_,
                       is_kid = is_kid_,
                       kid_shape = kid_shapes_)


