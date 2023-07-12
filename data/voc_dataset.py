import numpy as np
import os
import torch
import torchvision
import xml.etree.ElementTree as ET


VOC_CLASSES = (  # always index 0
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
)


class VOCAnnotationTransform:
    def __init__(self, class_to_index=None,
                       keep_difficult=False):
        self.class_to_index = class_to_index or dict((zip(VOC_CLASSES, range(len(VOC_CLASSES)))))
        self.keep_difficult = keep_difficult

    def __call__(self, xml_root):
        xml_size = xml_root.find('size')
        if not xml_size:
            raise ValueError("No size tag found in XML file.")
        img_height = float(xml_size.find('height').text.strip())
        img_width = float(xml_size.find('width').text.strip())

        res = []
        for xml_obj in xml_root.iter('object'):
            difficult = int(xml_obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            class_name = xml_obj.find('name').text.lower().strip()
            xml_bndbox = xml_obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(xml_bndbox.find(pt).text) - 1
                # Use relative coordinate
                cur_pt = cur_pt / img_width if i % 2 == 0 else cur_pt / img_height
                bbox.append(cur_pt)
            label_idx = self.class_to_index[class_name]
            bbox.append(label_idx + 1) # Reserve a label for background.
            res.append(bbox)
        return torch.Tensor(res)


class VOCDataset(torch.utils.data.Dataset):
    VOC_MEAN = (0.485, 0.456, 0.406) # Values from torch bench
    VOC_STD = (0.229, 0.224, 0.225)

    def __init__(self, split, root="data/VOC2012", apply_img_transform=True):
        if not (split == 'train' or split == 'val') or not os.path.exists(root):
            raise ValueError("Invalid split or root path.")

        self.root = root
        self.image_ids = []

        with open(os.path.join(root, "ImageSets", "Main", f"{split}.txt")) as f:
            for line in f.readlines():
                self.image_ids.append(line.strip())

        self._image_path = os.path.join(root, "JPEGImages", "%s.jpg")
        self._annot_path = os.path.join(root, "Annotations", "%s.xml")
        self._annot_transform = VOCAnnotationTransform(keep_difficult=False)
        self._image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((300, 300), antialias=True),
            torchvision.transforms.Normalize(mean=[0., 0., 0.],
                                             std=[255.0, 255.0, 255.0]),
        ])
        self._apply_img_transform = apply_img_transform

    def __getitem__(self, index):
        img_tensor = self._load_data(index)
        return img_tensor

    def __len__(self):
        return len(self.image_ids)

    def _load_data(self, index):
        img_id = self.image_ids[index]
        img = torchvision.io.read_image(self._image_path % img_id) # Channel-first
        img = self._image_transform(img.float())
        xml_root = ET.parse(self._annot_path % img_id).getroot()
        bboxes = self._annot_transform(xml_root)
        return img, bboxes


if __name__ == '__main__':
    dataset = VOCDataset(root="VOC2012")
    for i in range(10):
        img, bboxes = dataset[i]
        print(img.shape)
        print(bboxes)