import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.geometry import avg_rot, perspective_projection, rot6d_to_rotmat, rotmat_to_rot6d
from .refit import REFIT


class REFIT_MV(REFIT):
    def __init__(self, *args, **kwargs):
        super(REFIT_MV, self).__init__(*args, **kwargs)


    def forward(self, batch, iters=5):
        image  = batch['img']
        center = batch['center']
        scale  = batch['scale']
        img_focal = batch['img_focal']
        img_center = batch['img_center']
        cam_R = batch['cam_R']
        cam_t = batch['cam_t'] / 1000

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

        out['pred_shape'] = self.average_shape(out['pred_shape'])
        out['pred_pose'] = self.average_pose(out['pred_pose'], cam_R)
        # out['pred_cam'] = self.average_cam(out['pred_cam'], cam_R, cam_t, center, scale, img_focal, img_center)

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

            out['pred_shape'] = self.average_shape(out['pred_shape'])
            out['pred_pose'] = self.average_pose(out['pred_pose'], cam_R)
            # out['pred_cam'] = self.average_cam(out['pred_cam'], cam_R, cam_t, center, scale, img_focal, img_center)

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

        trans_full = self.get_trans(out['pred_cam'], center, scale, img_focal, img_center)
        out['trans_full'] = trans_full

        return out, iter_preds


    def average_shape(self, shape):
        bn = shape.size(0)
        avg_shape = shape.mean(dim=0, keepdim=True).repeat(bn,1)
        return avg_shape


    def average_pose(self, pose, cam_R):
        bn = pose.size(0)
        body_pose = pose[:, 6:]
        body_pose = body_pose.mean(dim=0, keepdim=True).repeat(bn, 1)

        global_pose = pose[:, :6]
        global_rotmat = rot6d_to_rotmat(global_pose)
        global_rotmat_w = torch.einsum('bij, bjk -> bik', cam_R.transpose(-1,-2), global_rotmat)
        global_rotmat_w = avg_rot(global_rotmat_w)
        global_rotmat = torch.einsum('bij, jk -> bik', cam_R, global_rotmat_w)

        # global_pose_s = rotmat_to_rot6d(global_rotmat)
        # scaling = (global_pose_s / global_pose).abs()
        # scaling, _ = torch.max(scaling, dim=1, keepdim=True)
        # global_pose = global_pose_s / scaling
        global_pose = rotmat_to_rot6d(global_rotmat)

        avg_pose = torch.concat([global_pose, body_pose], dim=1)
        return avg_pose


    def average_cam(self, cam, cam_R, cam_t, center, scale, img_focal, img_center):
        bn = cam.size(0)
        # fullframe translation w.r.t cam
        trans_full = self.get_trans(cam, center, scale, img_focal, img_center)
        trans_full = trans_full.squeeze()
        
        # translation in the world frame; then average
        n = cam_R.size(0)
        trans_w = torch.einsum('bij, bj -> bi', cam_R.transpose(-1, -2), trans_full - cam_t)
        trans_w = trans_w.mean(dim=0, keepdim=True).repeat(bn, 1)

        # full translation w.r.t cam
        trans_full = torch.einsum('bij, bj -> bi', cam_R, trans_w) + cam_t
        
        # convert fullframe trans to crop cam
        b = scale*200
        cx, cy = center[:,0], center[:,1]
        img_cx, img_cy = img_center[:,0], img_center[:,1] 

        tx_full, ty_full, tz_full = trans_full.unbind(-1)
        bs = 2*img_focal/tz_full

        tx = tx_full - 2*(cx-img_cx)/bs
        ty = ty_full - 2*(cy-img_cy)/bs
        s = bs / b
        avg_cam = torch.stack([s, tx, ty], dim=-1)
        return avg_cam


