"""
Instance-Segmentation-Projects
Nick Kaparinos
2021
"""

import time
from os import makedirs
import torch
from torch.utils.data import DataLoader
import wandb
from engine import train_one_epoch, evaluate
from utilities import *

if __name__ == '__main__':
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/balloon/' + time_stamp + '/'
    makedirs(LOG_DIR, exist_ok=True)

    # Datasets
    dataset = BalloonDataset(data_path='/home/nikos/Nikos/Projects/Sartorius-Cell-Instance-Segmentation/balloon/train/')

    indices = torch.randperm(len(dataset)).tolist()
    train_dataset = torch.utils.data.Subset(dataset, indices[:-10])
    test_dataset = torch.utils.data.Subset(dataset, indices[:10])
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True, num_workers=1,
                                                   prefetch_factor=1, collate_fn=collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                                                  prefetch_factor=1, collate_fn=collate_fn)

    # Wandb logging
    name = 'maskRCNN'
    config = dict()
    notes = ''
    wandb.init(project="balloon-instance-segmentation", entity="nickkaparinos", name=name, config=config, notes=notes,
               reinit=True)

    # Model
    model = get_mask_rcnn_model(num_classes=2).to(device=device)
    wandb.watch(model, log='all')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        # Train
        print(f'Training epoch {epoch}')
        model.train()
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=20)
        lr_scheduler.step()

        # Evaluate
        evaluate(model, test_dataloader, epoch, device=device)

    # Visualize predictions and ground truth
    model.eval()
    model = model.to(device)
    for image_idx, (images, targets) in enumerate(test_dataloader):
        image = list(image.to(device) for image in images)
        output = model(image)
        test_image = image[0].permute(1, 2, 0)
        visualise(image=test_image, annotations=output, log_dir=LOG_DIR, image_num=image_idx, mask_color='random',
                  thing_name='balloon')
        visualise(image=test_image, annotations=targets, log_dir=LOG_DIR, image_num=image_idx, mask_color='random',
                  thing_name='balloon', ground_truth=True)

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
