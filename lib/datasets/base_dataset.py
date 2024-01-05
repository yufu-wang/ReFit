import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize, ToTensor, Compose
import numpy as np
import cv2
from os.path import join
import joblib
import logging

from data_config import JOINT_REGRESSOR_H36M, PASCAL_OCCLUDERS
from lib.core import constants, config
from lib.utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
from lib.utils import rotation_conversions as geo
from lib.utils.geometry import perspective_projection, estimate_translation

from .coco_occlusion import occlude_with_pascal_objects
from lib.models.smpl import SMPL
from time import time

smpl = SMPL()
smpl_male = SMPL(gender='male')
smpl_female = SMPL(gender='female')

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, dataset, ignore_3d=False, use_augmentation=True, is_train=True,
                normalization=False, cropped=False, crop_size=224):
        super(BaseDataset, self).__init__()
        
        self.is_train = is_train

        self.dataset = dataset
        self.data = np.load(config.DATASET_FILES[is_train][dataset])

        self.img_dir = config.DATASET_FOLDERS[dataset]
        self.imgname = self.data['imgname'].astype(np.string_)
        self.normalization = normalization
        self.normalize_img = Compose([
                            ToTensor(),
                            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
                        ])


        self.J_regressor = torch.from_numpy(np.load(JOINT_REGRESSOR_H36M)).float()
        self.crop_size = crop_size

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        self.sc = 1.0
        
        # If False, do not do augmentation
        self.cropped = cropped
        self.use_augmentation = use_augmentation
        if use_augmentation:
            self.occluders = joblib.load(PASCAL_OCCLUDERS) 

        if self.cropped:
            self.orig_shape = self.data['orig_shape']

        # Get camera intrinsic, if available
        try:    
            self.img_focal = self.data['img_focal']
            self.img_center = self.data['img_center']
            self.has_camcalib = True
            print(dataset, 'has camera intrinsics')
        except KeyError:
            self.has_camcalib = False

        # Get camera extrinsic, if available
        try:    
            self.cam_R = self.data['cam_R']
            self.cam_t = self.data['cam_t']
            self.trans = self.data['trans']
            self.has_extrinsic = True
            print(dataset, 'has camera extrinsic')
        except KeyError:
            self.has_extrinsic = False
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))

        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
            print(dataset, 'has pose_3d')
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d:
            self.has_pose_3d = 0

        if 'coco' in dataset or 'mpii' in dataset:
            self.has_pose_3d = 0
            print('Not using pose3d for', dataset)
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)


        self.length = self.scale.shape[0]


    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)     # per channel pixel-noise
        rot = 0             # rotation
        sc = self.sc            # scaling
        occ = 0             # synthetic occlusion
        # if self.is_train and self.use_augmentation:
        if self.use_augmentation:
            OCCLUDE_PROB = 0.5
            FLIP_PROB = 0.5
            NOISE_FACTOR = 0.4
            ROT_FACTOR = 30
            SCALE_FACTOR = 0.25

            if np.random.uniform() <= OCCLUDE_PROB:
                occ = 1

            if np.random.uniform() <= FLIP_PROB:
                flip = 1

            if np.random.uniform() <= 0.0:
                rot = 0
            else:
                rot = min(2*ROT_FACTOR,
                      max(-2*ROT_FACTOR, np.random.randn()*ROT_FACTOR))
            
            sc = min(1+SCALE_FACTOR,
                 max(1-SCALE_FACTOR, np.random.randn()*SCALE_FACTOR+1))

            pn = np.random.uniform(1-NOISE_FACTOR, 1+NOISE_FACTOR, 3)

        return flip, pn, rot, sc, occ

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn, occ):
        """Process rgb image and do augmentation."""
        if not self.cropped:
            rgb_img = crop(rgb_img, center, scale, 
                        [self.crop_size, self.crop_size], rot=rot)

        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)

        # occlusion augmentation: PARE uses this.
        if occ:
            rgb_img = occlude_with_pascal_objects(rgb_img, self.occluders)
            

        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        
        return rgb_img.astype('uint8')

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                 [self.crop_size, self.crop_size], rot=r)

        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/self.crop_size - 1.

        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def center_processing(self, center, rot, flip, orig_shape):
        WH = orig_shape[::-1]

        if flip:
            rot = -rot
            center = center - WH/2
            center[0] = -center[0]
            center = center + WH/2
            
        R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot))],
                      [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot))]])

        aug_center = R @ (center - WH/2) + WH/2

        return aug_center


    def est_pose_3d(self, item):
        shape = item['betas'][np.newaxis]
        pose = item['pose'].reshape(24, 3)[np.newaxis]
        pose = geo.axis_angle_to_matrix(pose)

        gender = item['gender']
        if gender == 0:
            out = smpl_male(global_orient=pose[:, [0]], body_pose=pose[:, 1:], betas=shape)
        elif gender == 1:
            out = smpl_female(global_orient=pose[:, [0]], body_pose=pose[:, 1:], betas=shape)
        else:
            out = smpl(global_orient=pose[:, [0]], body_pose=pose[:, 1:], betas=shape)

        vertices = out.vertices[0]
        J_regressor = self.J_regressor

        if self.is_train:
            # for 3dpw during training, use smpl joints
            gt_keypoints_3d = out.joints[0, 25:]
            gt_verts = vertices
        else:
            # for 3dpw during testing, use regressed h36m joints
            gt_keypoints_3d = torch.matmul(J_regressor, vertices)

            gt_pelvis = gt_keypoints_3d[[0], :].clone()
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
            gt_verts = vertices - gt_pelvis

        conf = torch.ones([len(gt_keypoints_3d), 1])
        gt_keypoints_3d = torch.concat([gt_keypoints_3d, conf], axis=-1)

        return gt_keypoints_3d.numpy(), gt_verts.numpy()


    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc, occ = self.augm_params()
        item['flip'] = flip
        item['occ'] = occ
        item['rot'] = rot
        item['sc'] = sc
        
        # Load image
        imgname = str(self.imgname[index], encoding='utf-8')
        imgname = join(self.img_dir, imgname)

        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            logger.info(f"cv2 loading error image={imgname}")

        # bug
        if self.dataset=='bedlam' and self.img_center[index][0] == 360: # supposely a bedlam tall image 
            if img.shape[1] != 720:  # but the given image is not .......
                img = np.transpose(img, [1,0,2])[:,::-1,:].copy()

        orig_shape = np.array(img.shape)[:2]

        if self.cropped:
            orig_shape = self.orig_shape[index]

        # Get camera intrinsics
        if self.has_camcalib:
            item['img_focal'] = self.img_focal[index]
            item['img_center'] = self.img_center[index]
        else:
            item['img_focal'] = self.est_focal(orig_shape)
            item['img_center'] = self.est_center(orig_shape)

        # Get camera extrinsic
        ### only used for multiview h36m
        if self.has_extrinsic:
            item['cam_R'] = torch.from_numpy(self.cam_R[index]).float()
            item['cam_t'] = torch.from_numpy(self.cam_t[index]).float()
            item['trans'] = torch.from_numpy(self.trans[index]).float()


        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img = self.rgb_processing(img, center, sc*scale, rot, flip, pn, occ)
        if self.normalization:
            img = self.normalize_img(img)


        # Store unnormalize image
        item['img'] = img
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname

        # Get 3D joints for training, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get SMPL 3D joints for evaluation
        if self.is_train == False and 'mpi' not in self.dataset:
            item['gender'] = self.gender[index]
            S, gt_verts = self.est_pose_3d(item)
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
            item['gt_verts'] = gt_verts


        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()


        # Apply augmentation transforms to bbox center
        center = self.center_processing(center, rot, flip, orig_shape)
        

        item['scale'] = torch.tensor(sc * scale).float()
        item['center'] = torch.from_numpy(center).float()
        item['orig_shape'] = torch.from_numpy(orig_shape).float()
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        
        item['sample_index'] = index
        item['gender'] = self.gender[index]
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['dataset_name'] = self.dataset

        item['img_focal'] = torch.tensor(item['img_focal']).float()
        item['img_center'] = torch.from_numpy(item['img_center']).float()

        return item


    def __len__(self):
        return len(self.imgname)


    def est_focal(self, orig_shape):
        h, w = orig_shape
        focal = np.sqrt(h**2 + w**2)
        return focal

    def est_center(self, orig_shape):
        h, w = orig_shape
        center = np.array([w/2., h/2.])
        return center


