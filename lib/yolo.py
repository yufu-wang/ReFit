import cv2
import numpy as np
import torch
import matplotlib
current_backend = matplotlib.get_backend()

import sys
sys.path.insert(0,'yolov7')

from yolov7.models.experimental import attempt_load
from yolov7.utils.general import check_img_size
from yolov7.utils.datasets import LoadStreams, LoadImages, letterbox
from yolov7.utils.general import non_max_suppression, scale_coords
matplotlib.use(current_backend)

class Yolov7():
	def __init__(self, weights='data/pretrain/yolov7.pt', imgsz=640, device='cpu'):
		self.device = device
		self.model = attempt_load(weights, map_location=device)
		self.stride = int(self.model.stride.max())
		self.imgsz = check_img_size(imgsz, s=self.stride)
		self.model.eval()


	def __call__(self, img, classes=[0], conf=0.25, iou=0.45):
		# Input: img is loaded from cv2[:,:,::-1], aka, RGB
		#		 classes is class filter, eg [0] for only human
		# Output: pred is [x1, y1, x2, y2, conf, cls]

		imgsz = self.imgsz
		stride = self.stride

		img_d = letterbox(img, imgsz, stride)[0]
		img_d = img_d.transpose(2, 0, 1).copy()
		img_d = torch.from_numpy(img_d).float().unsqueeze(0) / 255.0
		img_d = img_d.to(self.device)

		with torch.no_grad():
			pred = self.model(img_d)[0]
			pred = non_max_suppression(pred, conf, iou, classes=classes, agnostic=False)[0]
			# (conf_thresh, iou_thresh) = conf, iou

		if len(pred):
			pred[:,:4] = scale_coords(img_d.shape[2:], pred[:, :4], img.shape).cpu()

		return pred





