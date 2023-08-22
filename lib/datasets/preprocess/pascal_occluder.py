import cv2
import joblib
import os
import os.path
import PIL.Image
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


pascal_voc_root_path = '/Users/yufu/vision_database/VOC2012'

occluders = []
structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
for annotation_path in tqdm(annotation_paths):
    xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
    is_segmented = (xml_root.find('segmented').text != '0')

    if not is_segmented:
        continue

    boxes = []
    for i_obj, obj in enumerate(xml_root.findall('object')):
        is_person = (obj.find('name').text == 'person')
        is_difficult = (obj.find('difficult').text != '0')
        is_truncated = (obj.find('truncated').text != '0')
        if not is_person and not is_difficult and not is_truncated:
            bndbox = obj.find('bndbox')
            box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append((i_obj, box))

    if not boxes:
        continue

    im_filename = xml_root.find('filename').text
    seg_filename = im_filename.replace('jpg', 'png')

    im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
    seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

    im = np.asarray(PIL.Image.open(im_path))
    labels = np.asarray(PIL.Image.open(seg_path))

    for i_obj, (xmin, ymin, xmax, ymax) in boxes:
        object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
        object_image = im[ymin:ymax, xmin:xmax]
        if cv2.countNonZero(object_mask) < 500:
            # Ignore small objects
            continue

        # Reduce the opacity of the mask along the border for smoother blending
        eroded = cv2.erode(object_mask, structuring_element)
        object_mask[eroded < object_mask] = 192
        object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

        # Downscale for efficiency
        object_with_mask = resize_by_factor(object_with_mask, 0.5)
        occluders.append(object_with_mask)

print('Saving pascal occluders')
joblib.dump(occluders, 'pascal_occluders.pkl', protocol=2)


