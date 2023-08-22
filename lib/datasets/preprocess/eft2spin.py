"""
Export EFT results (json) to SPIN style npz file
"""

import os
from os.path import join
import sys
import json
from tqdm import tqdm
import numpy as np
import torch
import cv2

from ...utils import rotation_conversions as geo


EFT2NPZ = {'COCO2014-All': 'coco_2014_eft.npz',
           'MPII': 'mpii_train_eft.npz'}

def EFT2Spin(eftfile, out_path='data/dataset_extras'):

    # scaleFactor = 1.2

    # open eft file
    with open(eftfile, 'r') as file:
        eft = json.load(file)
        meta = eft['meta']
        data = eft['data']
        dbname = meta['dbname']

    newfile = EFT2NPZ[dbname]
    newpath = join(out_path, newfile)

    # structs we need
    imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], []

    #additional 3D
    poses_ , shapes_, cams_, has_smpl_  = [], [] ,[], []


    for d in tqdm(data):
        imgname = d['imageName']
        center = d['bbox_center']
        scale = d['bbox_scale']

        keypoints = d['gt_keypoint_2d']
        pose = d['parm_pose']
        shape = d['parm_shape']
        cam = d['parm_cam']
        annotId = d['annotId']


        # To SPIN format
        if dbname == 'COCO2014-All':
            imgname = join('train2014', imgname)

        elif dbname == 'MPII':
            imgname = join('images', imgname)

        else:
            print('dataset not implemented')

        smpl_pose = geo.matrix_to_axis_angle(torch.tensor(pose)).flatten().numpy()
        smpl_shape = shape
        smpl_cam = cam

        openpose2d= keypoints[:25]        #25,3
        spin_smpl24= keypoints[25:]     #24,3

        #Save data
        imgnames_.append(imgname)
        centers_.append(center)
        scales_.append(scale)
        has_smpl_.append(1)
        poses_.append(smpl_pose)        #(72,)
        shapes_.append(smpl_shape)       #(10,)
        cams_.append(smpl_cam)
        openposes_.append(openpose2d)       #blank
        parts_.append(spin_smpl24)


    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    print(f"Save to {newpath}")
    np.savez(newpath, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       
                       openpose=openposes_,
                       part=parts_,

                       pose=poses_,
                       shape=shapes_,
                       #cam=cams_,
                       has_smpl=has_smpl_,)


if __name__ == '__main__':
    eftfile = 'data/eft/COCO2014-All-ver01.json'
    EFT2Spin(eftfile, out_path='data/dataset_extras')

