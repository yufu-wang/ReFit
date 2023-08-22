import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.imutils import flip_pose
from lib.utils import rotation_conversions as geo
from lib.utils.geometry import perspective_projection, rot6d_to_rotmat, rotmat_to_rot6d, avg_rot

from .hrnet_refit import hrnet_w32, hrnet_w48
from .utils import bilinear_sampler
from .multi_linear import MultiLinear
from .update import UpdateBlock, Regressor
from .smpl import SMPL


BN_MOMENTUM = 0.1

class REFIT(nn.Module):
    def __init__(self, ptype='marker', point_dim=1, radius=1,
                flow_dim=5, local_dim=256, hidden_dim=32, corr_layer=0,
                device='cpu', backbone='hrnet_w32', cfg=None, **kwargs):

        super(REFIT, self).__init__()
        self.device = device
        self.cfg = cfg
        self.crop_size = cfg.IMG_RES

        # SMPL
        self.smpl = SMPL()
        self.smpl_male = SMPL(gender='male')
        self.smpl_female = SMPL(gender='female')
        
        self.set_smpl_mean()
        if ptype == 'marker':
            self.ssm = self.smpl.ssm
            self.np=np=len(self.ssm)

        elif ptype == 'dense':
            self.ssm = self.smpl.dense
            self.np=np=len(self.ssm)


        # Model 
        self.radius = radius
        self.point_dim = point_dim
        self.flow_dim = flow_dim

        # Backbone
        if backbone == 'hrnet_w32':
            self.backbone = hrnet_w32(crop_size=self.crop_size)
            f_dim, z_dim = 480, 256
        elif backbone == 'hrnet_w48':
            self.backbone = hrnet_w48(crop_size=self.crop_size)
            f_dim, z_dim = 720, 384   

        # Initial and correlation layer
        self.init_layer = nn.Linear(z_dim, 26*hidden_dim)
        self.corr_layer = self._make_corr_layer(f_dim, point_dim*np, corr_layer)

        # Feedback module
        area = (2*radius+1)**2
        self.flow_layer = MultiLinear(np, area * point_dim, flow_dim)
        self.local_layer = nn.Linear(np * flow_dim, local_dim)
        self.masked_layer = nn.Dropout1d(p=cfg.TRAIN.MASKED_PROB)
        
        # Update module
        hidden_dim = hidden_dim
        input_dim = local_dim + 6*24 + 13 + 3

        self.update_block = UpdateBlock(input_dim, hidden_dim)
        self.regressor = Regressor(hidden_dim, hidden_dim, num_layer=cfg.MODEL.REG_LAYER)


    def forward(self, batch, iters=5, flip_test=False):
        image  = batch['img']
        center = batch['center']
        scale  = batch['scale']
        img_focal = batch['img_focal']
        img_center = batch['img_center']

        # estimate focal length, and bbox
        bbox_info = self.bbox_est(center, scale, img_focal, img_center)

        # backbone
        feature, z = self.backbone(image)

        BN = feature.shape[0]

        # initial estimate
        h = self.init_layer(z)
        h = torch.tanh(h)

        h = h.view(BN, 26, -1)
        hpose = h[:, :24]      # [bn, 24, h]
        hshape = h[:, 24]      # [bn, h]
        hcam = h[:, 25]        # [bn, h]

        # local feature network
        corr = self.corr_layer(feature)


        ####### initilization #######
        rotmat_preds  = [] 
        shape_preds = []
        cam_preds   = []
        j3d_preds = []
        j2d_preds = []

        d_pose, d_shape, d_cam = self.regressor(hpose, hshape, hcam, bbox_info)

        out = {}
        out['pred_cam'] = self.init_cam + d_cam
        out['pred_pose'] = self.init_pose + d_pose
        out['pred_shape'] = self.init_shape + d_shape
        out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(BN, 24, 3, 3)

        s_out = self.smpl.query(out)
        j3d = s_out.joints
        j2d = self.project(j3d, out['pred_cam'], center, scale, img_focal, img_center)

        rotmat_preds.append(out['pred_rotmat'].clone())
        shape_preds.append(out['pred_shape'].clone())
        cam_preds.append(out['pred_cam'].clone())
        j3d_preds.append(j3d.clone())
        j2d_preds.append(j2d.clone())

        masks = torch.ones([BN, self.np, self.flow_dim]).to(self.device)
        masks = self.masked_layer(masks)  # (BN, NP, C) randomly masked NP dimention
        
        ####### main LOOP #######
        hpose = torch.zeros_like(hpose).detach()
        hshape = torch.zeros_like(hshape).detach()
        hcam = torch.zeros_like(hcam).detach()

        for i in range(iters):
            cam = out['pred_cam'].detach()
            pose = out['pred_pose'].detach()
            shape = out['pred_shape'].detach()

            p3d = s_out.vertices[:, self.ssm]
            p2d = self.project(p3d, cam, center, scale, img_focal, img_center)
            p2d = p2d.detach()  # [bn, np, 2]

            # Local look up for the np markers (e.g. p2d in 224x224; coords in 56x56)
            coords = p2d / 4.
            loc = self.lookup(corr, coords)         #[bn, np, 7x7x1]
            loc = self.flow_layer(loc)              #[bn, np, 5]
            loc = (loc * masks).view(BN, -1)        #[bn, npx5] = [bn, 335]
            loc = self.local_layer(loc)             #[bn, 256]

            # update and regress
            loc = torch.cat([loc, bbox_info], dim=-1)

            hpose, hshape, hcam = self.update_block(hpose, hshape, hcam,
                                                    loc, pose, shape, cam)

            d_pose, d_shape, d_cam = self.regressor(hpose, hshape, hcam, bbox_info)


            out['pred_cam'] = cam + d_cam
            out['pred_pose'] = pose + d_pose
            out['pred_shape'] = shape + d_shape
            out['pred_rotmat'] = rot6d_to_rotmat(out['pred_pose']).reshape(BN, 24, 3, 3)
            
            s_out = self.smpl.query(out)
            j3d = s_out.joints
            j2d = self.project(j3d, out['pred_cam'], center, scale, img_focal, img_center)

            rotmat_preds.append(out['pred_rotmat'].clone())
            shape_preds.append(out['pred_shape'].clone())
            cam_preds.append(out['pred_cam'].clone())
            j3d_preds.append(j3d.clone())
            j2d_preds.append(j2d.clone())

        iter_preds = [rotmat_preds, shape_preds, cam_preds, j3d_preds, j2d_preds]
        #########################

        if flip_test:
            out_flip = self.pred_flip(batch, iters)
            out_ = self.flip_preds(out_flip)
            out = self.avg_preds(out, out_)

        trans_full = self.get_trans(out['pred_cam'], center, scale, img_focal, img_center)
        out['trans_full'] = trans_full
        
        return out, iter_preds


    def pred_flip(self, batch, iters):
        batch_flip = {}
        batch_flip['scale'] = batch['scale']
        batch_flip['img_focal'] = batch['img_focal']
        batch_flip['img_center'] = batch['img_center']
        
        WH = batch['orig_shape'].flip(dims=(1,))
        center = batch['center'] - WH/2
        center[:,0] = -center[:,0]
        center = center + WH/2
        batch_flip['center'] = center
        batch_flip['img'] = torch.flip(batch['img'], (3,)) # flip along x axis

        out, _ = self.forward(batch_flip, iters, flip_test=False)
        return out


    def flip_preds(self, out):
        rotmat = out['pred_rotmat']
        BN = len(rotmat)

        aa = geo.matrix_to_axis_angle(rotmat).reshape(BN, -1)
        aa = flip_pose(aa.transpose(0, 1)).transpose(0, 1)
        aa = aa.reshape(BN, -1, 3)
        rotmat = geo.axis_angle_to_matrix(aa)

        out['pred_rotmat'] = rotmat
        out['pred_pose'] = rotmat_to_rot6d(rotmat)
        return out


    def avg_preds(self, out, out_flip):
        rotmat = out['pred_rotmat']
        shape = out['pred_shape']

        rotmat_flip = out_flip['pred_rotmat']
        shape_flip = out_flip['pred_shape']

        pred_shape = (shape + shape_flip) / 2.
        pred_rotmat = torch.stack([rotmat, rotmat_flip])
        pred_rotmat = avg_rot(pred_rotmat)
        out['pred_rotmat'] = pred_rotmat
        out['pred_shape'] = pred_shape
        return out


    def lookup(self, corr, coords):
        r = self.radius
        h, w = corr.shape[-2:]
        device = corr.device

        h, w = corr.shape[-2:]   # (batch, j*c, 56, 56)
        bn, j = coords.shape[:2] # (batch, 24, 2)

        dx = torch.linspace(-r, r, 2*r+1, device=device)
        dy = torch.linspace(-r, r, 2*r+1, device=device)

        # lookup window
        centroid = coords.reshape(bn*j, 1, 1, 2)   # (bn*j, 1, 1, 2)

        delta = torch.stack(torch.meshgrid(dy, dx, indexing='ij'), axis=-1)
        delta = delta.view(1, 2*r+1, 2*r+1, 2)
        window = centroid + delta                  # (bn*j, 2r+1, 2r+1, 2)

        # feature map
        corr = corr                                # (bn, j*c, 56, 56)
        fmap = corr.view(bn*j, -1, h, w)           # (bn*j, c, 56, 56)
        feature = bilinear_sampler(fmap, window)   # (bn*j, c, 2r+1, 2r+1)
        feature = feature.view(bn, j, -1)          # (bn, j, c*(2r+1)*(2r+1))

        return feature


    def project(self, points, pred_cam, center, scale, img_focal, img_center, return_full=False):

        trans_full = self.get_trans(pred_cam, center, scale, img_focal, img_center)

        # Projection in full frame image coordinate
        points = points + trans_full
        points2d_full = perspective_projection(points, rotation=None, translation=None,
                        focal_length=img_focal, camera_center=img_center)

        # Adjust projected points to crop image coordinate
        # (s.t. 1. we can calculate loss in crop image easily
        #       2. we can query its pixel in the crop
        #  )
        b = scale * 200
        points2d = points2d_full - (center - b[:,None]/2)[:,None,:]
        points2d = points2d * (self.crop_size / b)[:,None,None]

        if return_full:
            return points2d_full, points2d
        else:
            return points2d


    def get_trans(self, pred_cam, center, scale, img_focal, img_center):
        b      = scale * 200
        cx, cy = center[:,0], center[:,1]            # center of crop
        s, tx, ty = pred_cam.unbind(-1)

        img_cx, img_cy = img_center[:,0], img_center[:,1]  # center of original image
        
        bs = b*s
        tx_full = tx + 2*(cx-img_cx)/bs
        ty_full = ty + 2*(cy-img_cy)/bs
        tz_full = 2*img_focal/bs

        trans_full = torch.stack([tx_full, ty_full, tz_full], dim=-1)
        trans_full = trans_full.unsqueeze(1)

        return trans_full


    def project_crop(self, points, pred_cam):
        crop_size = self.crop_size
        crop_center = crop_size / 2.

        s, tx, ty = pred_cam.split([1,1,1], dim=-1)
        points = points + torch.cat([tx, ty, 2*5000 / (s*crop_size)], dim=-1).unsqueeze(1)

        points2d = perspective_projection(points, rotation=None, translation=None,
                        focal_length=5000, camera_center=crop_center)
        return points2d


    def bbox_est(self, center, scale, img_focal, img_center):
        # Original image center
        img_cx, img_cy = img_center[:,0], img_center[:,1]

        # Implement CLIFF (Li et al.) bbox feature
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_cx, cy - img_cy, b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / img_focal.unsqueeze(-1) * 2.8 
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * img_focal) / (0.06 * img_focal)  

        return bbox_info


    def _make_corr_layer(self, in_dim, out_dim, num_layers=2):

        layers = []
        plane = 256

        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_dim, plane, kernel_size=3, stride=1,
                     padding=1, bias=False),
                nn.BatchNorm2d(plane, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
                ])

            in_dim = plane

        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, 
                                padding=0, bias=False))

        return nn.Sequential(*layers)


    def set_smpl_mean(self, ):
        SMPL_MEAN_PARAMS = 'data/smpl/smpl_mean_params.npz'

        mean_params = np.load(SMPL_MEAN_PARAMS)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)
        init_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.register_buffer('init_cam', init_cam)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


    def freeze_modules(self):
        frozen_modules = self.frozen_modules

        if frozen_modules is None:
            return

        for module in frozen_modules:
            if type(module) == torch.nn.parameter.Parameter:
                module.requires_grad = False
            else:
                module.eval()
                for p in module.parameters(): p.requires_grad=False

        return

    def unfreeze_modules(self, ):
        frozen_modules = self.frozen_modules

        if frozen_modules is None:
            return

        for module in frozen_modules:
            if type(module) == torch.nn.parameter.Parameter:
                module.requires_grad = True
            else:
                module.train()
                for p in module.parameters(): p.requires_grad=True

        self.frozen_modules = None

        return


        