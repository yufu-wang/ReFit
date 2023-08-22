"""
Export AGORA to SPIN style npz file
"""

import numpy as np
import torch
import os
import pickle
from os.path import join
from tqdm import tqdm
from glob import glob

from lib.utils import rotation_conversions as geo
from lib.models.smpl import SMPL
from lib.utils.geometry import perspective_projection



def get_2d_3d_joints(anns, i):
    # Generate 2D and 3D joints
    pose = anns['pose_cam'][i].reshape(1, -1, 3)
    shape = anns['shape'][i].reshape(1, -1)
    trans = anns['trans_cam'][i].reshape(1, -1)

    trans = anns['trans_cam'][i] + anns['cam_ext'][i][:3,3]
    trans = trans.reshape(1, -1)

    pose = torch.tensor(pose).float()
    shape = torch.tensor(shape).float()
    trans = torch.tensor(trans).float()
    rotmat = geo.axis_angle_to_matrix(pose)

    output = smpl(global_orient = rotmat[:,[0]],
                  body_pose = rotmat[:,1:],
                  betas = shape[:,:10])
    verts = output.vertices[0] + trans
    joints = output.joints[0] + trans

    cam_int = anns['cam_int'][i]
    focal = (cam_int[0,0] + cam_int[1,1]) / 2
    imgcenter = torch.tensor(cam_int[:2, 2])

    j3d = joints[25:]
    pelvis3d = j3d[[14]]
    j2d = perspective_projection(j3d[None], None, None, focal, imgcenter)[0]

    joints3d = (j3d - pelvis3d).numpy()
    joints2d = j2d.numpy()

    joints3d = np.concatenate([joints3d, np.ones([24, 1])], axis=1)
    joints2d = np.concatenate([joints2d, np.ones([24, 1])], axis=1)

    return joints3d, joints2d


def convert_bedlam(file):
    seq = os.path.basename(file).replace('.npz','')
    anns = dict(np.load(file))
    total = len(anns['imgname'])

    for i in tqdm(range(total)):
        imgname = join(seq, 'png', anns['imgname'][i])
        pose = anns['pose_cam'][i]
        shape = anns['shape'][i,:10]

        cam_ext = anns['cam_ext'][i]
        cam_int = anns['cam_int'][i]
        focal = (cam_int[0,0] + cam_int[1,1]) / 2.
        img_center = cam_int[:2, 2]

        joints3d, joints2d = get_2d_3d_joints(anns, i) 

        # center & scale
        scale = anns['scale'][i] * scaleFactor
        center = anns['center'][i]
        if scale <= 0.3:
            continue

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


if __name__ == '__main__':
    # ROOT
    root = '/Users/yufu/vision_database/bedlam'
    files = glob(root + '/all_npz_12_smpl_training/*.npz')
    smpl = SMPL()

    # scale factor
    scaleFactor = 1.12

    # structs we use
    imgnames_, scales_, centers_, S_, parts_ = [], [], [], [], []
    poses_, shapes_ = [], []
    img_focals_, img_centers_ = [], []

    # each ann file
    for file in tqdm(files):
        # we only have images for this anns available on mac
        # if os.path.basename(file) != '20221010_3-10_500_batch01hand_zoom_suburb_d_6fps.npz':
        #     continue
        
        convert_bedlam(file)


    out_file = 'data/dataset_extras/bedlam_train.npz'
    np.savez(out_file, imgname = imgnames_,
                       center = centers_,
                       scale = scales_,
                       S = S_,
                       part = parts_,
                       pose = poses_,
                       shape = shapes_,
                       img_focal = img_focals_,
                       img_center = img_centers_)


