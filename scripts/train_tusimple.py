#!/usr/bin/env python3

import time
import os
import sys
import numpy as np
import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from LaneNet import (
    build_optimizer,
    build_lr_scheduler,
)
from LaneNet.datasets import tuSimpleDataset as Dataset
from LaneNet.transforms import Rescale
from LaneNet.losses import compute_loss
from LaneNet.models import LaneNet
from LaneNet.utils import parse_args, AverageMeter

from .test_tusimple import test


def save_model(save_path, epoch, model):
    save_name = os.path.join(save_path, f'{epoch}_checkpoint.pth')
    torch.save(model, save_name)
    print("model is saved: {}".format(save_name))


def main():
    args = parse_args()

    save_path = args.save

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    train_dataset_file = os.path.join(args.dataset, 'train.txt')
    val_dataset_file = os.path.join(args.dataset, 'val.txt')

    train_dataset = Dataset(
        dataset_path=train_dataset_file,
        mode='train',
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    if args.val:
        val_dataset = Dataset(
            dataset_path=val_dataset_file,
            mode='test',
        )
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    model = LaneNet()
    model.

    optimizer = build_optimizer(model, optim='adam', lr=args.lr)
    lr_scheduler = build_lr_scheduler(optimizer, lr_scheduler='single_step', stepsize=50, gamma=0.1)
    print(f"{args.epochs} epochs {len(train_dataset)} training samples\n")

    for epoch in range(0, args.epochs):
        print(f"Epoch {epoch}")
        train_iou = train(train_loader, model, optimizer, epoch)
        if args.val:
            val_iou = test(val_loader, model, epoch)
        if (epoch + 1) % 5 == 0:
            save_model(save_path, epoch, model)

        print(f"Train IoU : {train_iou}")
        if args.val:
            print(f"Val IoU : {val_iou}")


if __name__ == '__main__':
    main()