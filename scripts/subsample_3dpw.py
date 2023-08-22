import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from data_config import PW3D_ROOT
from lib.datasets.base_dataset import BaseDataset


db = BaseDataset('3dpw_test', is_train=False, use_augmentation=False, normalization=False, 
                cropped=False, crop_size=256)


# new ann format for crops
data = np.load('data/dataset_extras/3dpw_test.npz', 'rw')
data = dict(data)
data['imgname'] = data['imgname'].astype('<U80')
orig_shape = []

for i, n in enumerate(data['imgname']):

    # get the original image shape
    img_path = os.path.join(PW3D_ROOT, n)
    imgfile = Image.open(img_path)
    
    shape = [imgfile.height, imgfile.width]
    orig_shape.append(shape)

    # change name
    newname = n.replace('imageFiles', 'imageCrops')
    newname = newname.replace('.jpg', '_{}.jpg'.format(i))
    data['imgname'][i] = newname

orig_shape = np.array(orig_shape)
data['orig_shape'] = orig_shape


# create a small subset
np.random.seed(0)
l = len(data['imgname'])
samples = np.random.permutation(l)[:3000]
samples = np.sort(samples)

for k in data:
    data[k] = data[k][samples]
np.savez('data/dataset_extras/3dpw_test_sub.npz', **data)



# save crops
for i in tqdm(samples):
    item = db[i]
    
    img = item['img']
    imgname = item['imgname']
    newname = imgname.replace('imageFiles', 'imageCrops')
    newname = newname.replace('.jpg', '_{}.jpg'.format(i))
    
    newdir = os.path.dirname(newname)
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    cv2.imwrite(newname, img[:,:,[2,1,0]])


