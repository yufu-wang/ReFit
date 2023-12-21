import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob

from lib.core.config import update_cfg
from lib.yolo import Yolov7
from lib import get_model
from lib.datasets.detect_dataset import DetectDataset
from lib.renderer.renderer_img import Renderer as Renderer_img
from pytorch3d.transforms import matrix_to_axis_angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, default='data/examples/ubc_examples')
    parser.add_argument("--cfg", type=str, default='configs/config.yaml')
    parser.add_argument("--ckpt",  type=str, default='data/pretrain/refit_all/checkpoint_best.pth.tar')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--viz_results", action='store_true')
    args = parser.parse_args()

    # image folder
    root = args.imagedir
    imgpaths = sorted(glob(f'{root}/*.png')) + sorted(glob(f'{root}/*.jpg'))

    # Yolo model
    DEVICE = args.device
    yolo = Yolov7(device=DEVICE, weights='data/pretrain/yolov7-e6e.pt', imgsz=640)

    # ReFit
    cfg = update_cfg(args.cfg)
    cfg.DEVICE = DEVICE
    
    model = get_model(cfg).to(DEVICE)
    state_dict = torch.load(args.ckpt, map_location=cfg.DEVICE)
    _ = model.load_state_dict(state_dict['model'], strict=False)
    _ = model.eval()

    # Rendering
    renderer_img = Renderer_img(model.smpl.faces, color=(0.40,  0.60,  0.9, 1.0))

    # Run on folder
    smpl_shape = []
    smpl_pose = []
    smpl_trans = []
    img_focal = []
    img_center = []

    count = 0
    for imgpath in tqdm(imgpaths):
        img = cv2.imread(imgpath)[:,:,::-1].copy()

        ### --- Detection ---
        with torch.no_grad():
            boxes = yolo(img, conf=0.50, iou=0.45)
            boxes = boxes.cpu().numpy()
            if len(boxes) > 1:
                valid = boxes[:, 2:4].max(axis=1).argmax()
                boxes = boxes[[valid]]
            
        db = DetectDataset(img, boxes, dilate=1.2)
        dataloader = torch.utils.data.DataLoader(db, batch_size=8, shuffle=False, num_workers=0)

        ### --- ReFit --- 
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items() if type(v)==torch.Tensor}
            with torch.no_grad():
                out, preds = model(batch, iters=5)
                s_out = model.smpl.query(out)
                verts = s_out.vertices + out['trans_full']

                smpl_pose.append(out['pred_rotmat'])
                smpl_shape.append(out['pred_shape'])
                smpl_trans.append(out['trans_full'])
        
        img_focal.append(db.img_focal)
        img_center.append(db.img_center)

        ### --- Render ---
        if args.viz_results:
            savefolder = 'results'
            os.makedirs(savefolder, exist_ok=True)

            img_render = renderer_img(verts, [0,0,0], img)
            new_name = f'{savefolder}/{os.path.basename(imgpath)}'
            cv2.imwrite(new_name, img_render[:,:,::-1].copy())

    # Save pose results for GART
    smpl_pose = torch.cat(smpl_pose)
    smpl_shape = torch.cat(smpl_shape)
    smpl_trans = torch.cat(smpl_trans)
    smpl_pose = matrix_to_axis_angle(smpl_pose)

    pose_dict = {'betas': smpl_shape.mean(0),
                'global_orient': smpl_pose[:,0],
                'body_pose': smpl_pose[:,1:].reshape(-1,69),
                'transl': smpl_trans.squeeze(1)}
    np.savez_compressed("poses_optimized.npz", **pose_dict)


    focal = np.mean(np.stack(img_focal))
    center = np.mean(np.stack(img_center), axis=0)

    K = np.eye(3)
    K[0, 0], K[1, 1] = focal, focal
    K[0, 2], K[1, 2] = center[0], center[1]
    cam_dict = {
        "intrinsic": K,
        "extrinsic": np.eye(4),
        "height": img.shape[0],
        "width": img.shape[1],
    }
    np.savez_compressed("cameras.npz", **cam_dict)
    













