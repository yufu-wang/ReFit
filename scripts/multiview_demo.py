import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from glob import glob
from torch.utils.data import default_collate

from lib.core.config import update_cfg
from lib import get_model
from lib.datasets.base_dataset import BaseDataset
from lib.renderer.renderer_img import Renderer as Renderer_img
from pytorch3d.transforms import matrix_to_axis_angle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='configs/config.yaml')
    parser.add_argument("--ckpt",  type=str, default='data/pretrain/refit_all/checkpoint_best.pth.tar')
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--viz_results", action='store_true')
    args = parser.parse_args()

    # multiview examples
    db = BaseDataset('multiview_examples', is_train=False, use_augmentation=False, 
                     normalization=True, cropped=False, crop_size=256)

    # refit
    DEVICE = args.device
    cfg = update_cfg(args.cfg)
    cfg.DEVICE = DEVICE
    cfg.MODEL.VERSION = 'mv'
    
    model = get_model(cfg).to(DEVICE)
    state_dict = torch.load(args.ckpt, map_location=cfg.DEVICE)
    _ = model.load_state_dict(state_dict['model'], strict=False)
    _ = model.eval()

    # Rendering
    renderer_img = Renderer_img(model.smpl.faces, color=(0.40,  0.60,  0.9, 1.0))

    # Load 4 views
    items = []
    for i in range(4):
        item = db[i]
        items.append(item)
    batch = default_collate(items)
    for k,v in batch.items():
        if type(v)==torch.Tensor:
            batch[k] = v.float().to(DEVICE)

    # multiview refitex
    with torch.no_grad():
        out, iter_preds = model(batch, 10)
        smpl_out = model.smpl.query(out)

    for k in range(4):
        imgfile = batch['imgname'][k]
        img_full = cv2.imread(imgfile)[:,:,::-1]

        vert = smpl_out.vertices[k]
        trans = out['trans_full'][k]
        vert_full = (vert + trans).cpu()

        focal = batch['img_focal'][k]
        center = batch['img_center'][k]
        img_render = renderer_img(vert_full, [0,0,0], img_full, focal, center)

        os.makedirs(f'mv_refit', exist_ok=True)
        cv2.imwrite(f'mv_refit/img_{i}_{k}.png', img_full[:,:,::-1])
        cv2.imwrite(f'mv_refit/mesh_{i}_{k}.png', img_render[:,:,::-1])














