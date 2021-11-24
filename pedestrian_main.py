"""
Balloon-Pedestrian-Instance-Segmentation
Kaggle competition
Nick Kaparinos
2021
"""

import utils
from utilities import *
from engine import *
import torch
import time
from os import makedirs

if __name__ == "__main__":
    start = time.perf_counter()
    seed = 0
    set_all_seeds(seed=seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(f"Using device: {device}")

    # Log directory
    time_stamp = str(time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()))
    LOG_DIR = 'logs/balloon/' + time_stamp + '/'
    makedirs(LOG_DIR, exist_ok=True)

    # Datasets
    dataset = PedestrianDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PedestrianDataset('PennFudanPed', get_transform(train=False))

    # Dataloaders
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:20])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2,
                                                   collate_fn=utils.collate_fn)
    test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2,
                                                  collate_fn=utils.collate_fn)
    # # Wandb logging
    # name = 'maskRCNN'
    # config = dict()
    # notes = ''
    # wandb.init(project="pedestrian-instance-segmentation", entity="nickkaparinos", name=name, config=config,
    #            notes=notes, reinit=True)

    # Model
    model = get_mask_rcnn_model(num_classes=2).to(device)
    # wandb.watch(model, log='all')
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Training and Evaluation
    num_epochs = 1
    for epoch in range(num_epochs):
        # Training
        print(f'Training epoch {epoch}')
        model.train()
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=20)
        lr_scheduler.step()

        # Evaluation
        # evaluate(model, test_dataloader, device=device)

    # Visualize
    model.eval()
    test_image = cv2.imread(
        '/home/nikos/Nikos/Projects/Baloon-Pedestrian-Instance-Segmentation/PennFudanPed/PNGImages/FudanPed00001.png')
    test_image = torch.tensor(test_image, dtype=torch.float32) / 255
    test_image = test_image.permute(2, 0, 1)
    model.eval()
    output = model([test_image.to(device)])
    test_image = test_image.permute(1, 2, 0)
    # visualise_v2(img=test_image * 255, output=output, thing_classes=['pedestrian'])
    visualise(image=test_image, output=output, image_num=0,mask_color='random',thing_name='pedestrian')

    # Execution Time
    end = time.perf_counter()
    print(f"\nExecution time = {end - start:.2f} second(s)")
