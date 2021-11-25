"""
Instance-Segmentation-Projects
Nick Kaparinos
2021
"""

import datetime
import errno
import json
import os
import random
import time
from collections import defaultdict, deque
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image, ImageDraw
from detectron2.data import Metadata
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


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
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = list(sorted(os.listdir(os.path.join(data_path, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(data_path, "PedMasks"))))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load images and masks
        img_path = os.path.join(self.data_path, "PNGImages", self.images[idx])
        mask_path = os.path.join(self.data_path, "PedMasks", self.masks[idx])
        img = cv2.imread(img_path)
        img = torch.tensor(img) / 255
        img = img.permute(2, 0, 1)

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

        return img, target


def get_mask_rcnn_model(num_classes):
    """ Build and return mask RCNN model """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=100)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model


def visualise(image, annotations, log_dir, image_num=0, mask_color=(0, 255, 0), thing_name='thing', score_threshold=0.5,
              scale=1, ground_truth=False) -> None:
    """
    Visualisation of maskRCNN instance segmentation predictions using detectron2
    :param image: (H,W,C) image
    :param annotations: pytorch maskRCNN output or ground truth
    :param image_num: image number
    :param mask_color: color of the segmentation mask tuple (int,int,int) or 'random'
    :param thing_name: name of the thing class
    :param score_threshold: threshold of the prediction score in order to be visualised
    :param ground_truth: boolean, whether annotations are ground truth
    """
    # Metadata
    metadata = Metadata(name=thing_name, thing_classes=[thing_name])
    if mask_color != 'random':
        metadata.set(thing_colors=[mask_color])

    # Convert pytorch output to detectron2 output
    predictions = {}
    predictions['pred_boxes'] = annotations[0]['boxes'].detach().to('cpu').numpy()
    if ground_truth:
        N = annotations[0]['boxes'].shape[0]
        predictions['scores'] = torch.ones(N, )
    else:
        predictions['scores'] = annotations[0]['scores'].to('cpu')
    predictions['pred_classes'] = annotations[0]['labels'].to(
        'cpu') - 1  # -1 because detectron2 needs the thing class index
    predictions['pred_masks'] = torch.squeeze(annotations[0]['masks'].to('cpu'), 1) > 0.5

    # Remove instance predictions with score lower than threshold
    predictions['pred_boxes'] = predictions['pred_boxes'][predictions['scores'].detach().numpy() > score_threshold]
    predictions['pred_classes'] = predictions['pred_classes'][predictions['scores'] > score_threshold]
    predictions['pred_masks'] = predictions['pred_masks'][predictions['scores'] > score_threshold]
    predictions['scores'] = predictions['scores'][predictions['scores'] > score_threshold]
    prediction_instances = Instances(image_size=(image.shape[0], image.shape[1]), **predictions)

    # Visualise
    image = image.detach().cpu().numpy()
    image_cpy = image.copy()
    visualizer = Visualizer(image[:, :, ::-1] * 255, metadata=metadata, scale=scale,
                            instance_mode=ColorMode.SEGMENTATION)
    out = visualizer.draw_instance_predictions(prediction_instances)
    image_segm = out.get_image()[:, :, ::-1]

    image_name = f'image_{image_num}'
    # cv2.imshow(image_name, image_cpy)
    # cv2.waitKey(0)
    cv2.imwrite(log_dir + image_name + '.jpg', image_cpy * 255)
    if ground_truth:
        image_name += '_ground_truth'
    image_name += '_segmentation'
    # cv2.imshow(image_name, image_segm)
    # cv2.waitKey(0)
    cv2.imwrite(log_dir + image_name + '.jpg', image_segm)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def collate_fn(batch):
    return tuple(zip(*batch))


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.inference_mode():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
