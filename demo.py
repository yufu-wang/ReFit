import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from glob import glob

from lib.core.config import parse_args
from lib import get_model
from lib.renderer.renderer_img import Renderer as Renderer_img

from lib.datasets.detect_dataset import DetectDataset
from lib.models.smpl import SMPL
from lib.yolo import Yolov7


# Yolo model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = Yolov7(device=DEVICE, weights='data/pretrain/yolov7-e6e.pt', imgsz=1281)

# ReFit
args = ['--cfg', 'configs/config.yaml']
cfg = parse_args(args)
cfg.DEVICE = DEVICE

model = get_model(cfg).to(DEVICE)
checkpoint = 'data/pretrain/refit_all/checkpoint_best.pth.tar'
state_dict = torch.load(checkpoint, map_location=cfg.DEVICE)
_ = model.load_state_dict(state_dict['model'], strict=False)
_ = model.eval()
print('Loaded checkpoint:', checkpoint)

# Rendering
smpl = SMPL()
renderer_img = Renderer_img(smpl.faces, color=(0.40,  0.60,  0.9, 1.0))

# Example image
imgfiles = sorted(glob('data/examples/*'))
for imgname in tqdm(imgfiles):
    img = cv2.imread(imgname)[:,:,::-1].copy()

    ### --- Detection ---
    with torch.no_grad():
        boxes = yolo(img, conf=0.50, iou=0.45)
        boxes = boxes.cpu().numpy()
        
    db = DetectDataset(img, boxes)
    dataloader = torch.utils.data.DataLoader(db, batch_size=8, shuffle=False, num_workers=0)

    ### --- ReFit ---
    vert_all = []
    for batch in dataloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items() if type(v)==torch.Tensor}
        with torch.no_grad():
            out, preds = model(batch, iters=5)
            s_out = model.smpl.query(out)
            vertices = s_out.vertices

        vert_full = vertices + out['trans_full']
        vert_all.append(vert_full)
        
    vert_all = torch.cat(vert_all).cpu()

    ### --- Render ---
    img_render = renderer_img(vert_all, [0,0,0], img)
    new_name = os.path.basename(imgname).replace('.png', '_result.png').replace('.jpg', '_result.jpg')
    cv2.imwrite(new_name, img_render[:,:,::-1].copy())
    print('Done')










