"""
Sartorius-Cell-Instance-Segmentation
Kaggle competition
Nick Kaparinos
2021
"""

from utilities import *
import time
import torch
import wandb
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
from tqdm import tqdm
from os import makedirs
import random
import logging
import sys
import cv2

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    if debugging:
        print("Debugging!")
    print(f"Using device: {device}")

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/balloon/' + time_stamp + '/'
    makedirs(LOG_DIR, exist_ok=True)

    # Datasets
    indices = [i for i in range(8)]
    dataset = BalloonDataset(data_path='/home/nikos/Nikos/Projects/Sartorius-Cell-Instance-Segmentation/balloon/train/')

    indices = [i for i in range(len(dataset))]  # torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:20])
    test_dataset = torch.utils.data.Subset(dataset, indices[:10])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                   prefetch_factor=1, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  prefetch_factor=1, collate_fn=collate_fn)

    # Wandb logging
    # name = 'maskRCNN'
    # config = dict()
    # notes = ''
    # wandb.init(project="balloon-instance-segmentation", entity="nickkaparinos", name=name, config=config, notes=notes,
    #            reinit=True)

    # Model
    model = get_mask_rcnn_model(num_classes=2).to(device=device)
    # wandb.watch(model, log='all')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 2
    for epoch in range(num_epochs):
        # Train
        print(f'Training epoch {epoch}')
        model.train()
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=20)
        lr_scheduler.step()

        # Evaluate
        # evaluate(model, test_dataloader, device=device)

    # Visualize
    model.eval()
    test_image = cv2.imread(
        '/home/nikos/Nikos/Projects/Baloon-Pedestrian-Instance-Segmentation/balloon/train/34020010494_e5cb88e1c4_k.jpg')
    test_image = torch.tensor(test_image, dtype=torch.float32) / 255
    test_image = test_image.permute(2, 0, 1)
    model.eval()
    torch.cuda.synchronize(device=device)
    torch.cuda.empty_cache()
    device = 'cpu'
    model = model.to(device)
    output = model([test_image.to(device)])
    test_image = test_image.permute(1, 2, 0)
    visualise(image=test_image, output=output, image_num=0, mask_color='random', thing_name='balloon',
              score_threshold=0.4, scale=0.5)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
