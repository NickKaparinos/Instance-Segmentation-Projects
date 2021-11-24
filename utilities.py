"""
Balloon-Pedestrian-Instance-Segmentation
Kaggle competition
Nick Kaparinos
2021
"""

import os
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import Metadata
from detectron2.structures.instances import Instances
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
from pycocotools.mask import encode
import torch
import json
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import cv2

debugging = True


class BalloonDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        json_file = data_path + 'via_region_data.json'
        image_list = os.listdir(data_path)
        image_list = [i for i in image_list if '.jpg' in i]
        self.image_list = image_list

        with open(json_file) as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        for idx, image_dict in enumerate(self.annotations.values()):
            if idx != index:
                continue
            img_array = cv2.imread(self.data_path + image_dict['filename'])

            N = len(image_dict['regions'])
            H = img_array.shape[0]
            W = img_array.shape[1]
            boxes = torch.ones((N, 4), dtype=torch.float32)
            masks = torch.zeros((N, H, W), dtype=torch.uint8)
            for index2, (_, instance_dict) in enumerate(image_dict['regions'].items()):
                # Mask
                x_points = instance_dict['shape_attributes']['all_points_x']
                y_points = instance_dict['shape_attributes']['all_points_y']
                points = [(i, j) for i, j in zip(x_points, y_points)]
                img = Image.new('L', (W, H), 0)
                ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
                mask = np.array(img)
                masks[index2] = torch.tensor(mask)

                # Bbox
                bbox = [np.min(x_points), np.min(y_points), np.max(x_points), np.max(y_points)]
                boxes[index2] = torch.tensor(bbox)
            labels = torch.ones((N), dtype=torch.int64)
            iscrowd = torch.zeros((N,), dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            image_id = torch.tensor([self.image_list.index(image_dict['filename'])])
            target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': image_id, 'area': area,
                      'iscrowd': iscrowd}
            break
        img_array = torch.tensor(img_array, dtype=torch.float32) / 255
        img_array = img_array.permute(2, 0, 1)
        return img_array, target


class PedestrianDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.images = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.images[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]

        # Bbox
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'masks': masks, 'image_id': image_id, 'area': area,
                  'iscrowd': iscrowd}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target



def get_mask_rcnn_model(num_classes):
    """ Build and return rcnn model """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def visualise(image, output, image_num=0, mask_color=(0, 255, 0), thing_name='thing', score_threshold=0.5, scale=1) -> None:
    """
    Visualisation of maskRCNN instance segmentation predictions using detectron2
    :param image: (H,W,C) image
    :param output: pytorch maskRCNN output
    :param image_num: image number
    :param mask_color: color of the segmentation mask tuple (int,int,int) or 'random'
    :param thing_name: name of the thing class
    :param score_threshold: threshold of the prediction score in order to be visualised
    """
    # Metadata
    metadata = Metadata(name=thing_name, thing_classes=[thing_name])
    if mask_color != 'random':
        metadata.set(thing_colors=[mask_color])

    # Convert pytorch output to detectron2 output
    predictions = {}
    predictions['pred_boxes'] = output[0]['boxes'].detach().to('cpu').numpy()
    predictions['scores'] = output[0]['scores'].to('cpu')
    predictions['pred_classes'] = output[0]['labels'].to('cpu') - 1  # -1 because detectron2 needs the thing class index
    predictions['pred_masks'] = torch.squeeze(output[0]['masks'].to('cpu'), 1) > 0.5

    # Remove instance predictions with score lower than threshold
    predictions['pred_boxes'] = predictions['pred_boxes'][predictions['scores'] > score_threshold]
    predictions['pred_classes'] = predictions['pred_classes'][predictions['scores'] > score_threshold]
    predictions['pred_masks'] = predictions['pred_masks'][predictions['scores'] > score_threshold]
    predictions['scores'] = predictions['scores'][predictions['scores'] > score_threshold]
    prediction_instances = Instances(image_size=(image.shape[0], image.shape[1]), **predictions)

    # Visualise
    image = image.numpy()
    image_cpy = image.copy()
    visualizer = Visualizer(image[:, :, ::-1]*255, metadata=metadata, scale=scale, instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_instance_predictions(prediction_instances)
    temp = out.get_image()[:, :, ::-1]


    cv2.imshow(f'test_image_{image_num}', image_cpy)
    cv2.waitKey(0)
    cv2.imshow(f'test_image_{image_num}_segmentation', temp)
    cv2.waitKey(0)
    return

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    return tuple(zip(*batch))
