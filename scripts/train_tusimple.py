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

# might want this in the transform as well
VGG_MEAN = [103.939, 116.779, 123.68]
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compose_img(image_data, out, binary_label, pix_embedding, instance_label, i):
    val_gt = (image_data[i].cpu().numpy().transpose(1, 2, 0) + VGG_MEAN).astype(np.uint8)
    val_pred = out[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_label = binary_label[i].squeeze(0).cpu().numpy().transpose(0, 1) * 255
    val_out = np.zeros((val_pred.shape[0], val_pred.shape[1], 3), dtype=np.uint8)
    val_out[:, :, 0] = val_pred
    val_out[:, :, 1] = val_label
    val_gt[val_out == 255] = 255
    # epsilon = 1e-5
    # pix_embedding = pix_embedding[i].data.cpu().numpy()
    # pix_vec = pix_embedding / (np.sum(pix_embedding, axis=0, keepdims=True) + epsilon) * 255
    # pix_vec = np.round(pix_vec).astype(np.uint8).transpose(1, 2, 0)
    # ins_label = instance_label[i].data.cpu().numpy().transpose(0, 1)
    # ins_label = np.repeat(np.expand_dims(ins_label, -1), 3, -1)
    # val_img = np.concatenate((val_gt, pix_vec, ins_label), axis=0)
    # val_img = np.concatenate((val_gt, pix_vec), axis=0)
    # return val_img
    return val_gt


def train(
    train_loader,
    model,
    optimizer,
    epoch
):
    batch_time = AverageMeter()
    mean_iou = AverageMeter()
    total_losses = AverageMeter()
    binary_losses = AverageMeter()
    instance_losses = AverageMeter()
    end = time.time()
    step = 0

    for batch_idx, batch in enumerate(iter(train_loader)):
        step += 1
        image_data = batch[0].to(DEVICE)
        binary_label = batch[1].type(torch.LongTensor).to(DEVICE)
        instance_label = batch[2].type(torch.FloatTensor).to(DEVICE)

        # forward pass
        net_output = model(image_data)

        # compute loss
        total_loss, binary_loss, instance_loss, out, train_iou = compute_loss(net_output, binary_label, instance_label)

        # update loss in AverageMeter instance
        total_losses.update(total_loss.item(), image_data.size()[0])
        binary_losses.update(binary_loss.item(), image_data.size()[0])
        instance_losses.update(instance_loss.item(), image_data.size()[0])
        mean_iou.update(train_iou, image_data.size()[0])

        # reset gradients
        optimizer.zero_grad()

        # backpropagate
        total_loss.backward()

        # update weights
        optimizer.step()

        # update batch time
        batch_time.update(time.time() - end)
        end = time.time()

        if step % 500 == 0:
            print(
                "Epoch {ep} Step {st} |({batch}/{size})| ETA: {et:.2f}|Total loss:{tot:.5f}|Binary loss:{bin:.5f}|Instance loss:{ins:.5f}|IoU:{iou:.5f}".format(
                    ep=epoch + 1,
                    st=step,
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    et=batch_time.val,
                    tot=total_losses.avg,
                    bin=binary_losses.avg,
                    ins=instance_losses.avg,
                    iou=train_iou,
                ))
            sys.stdout.flush()
            train_img_list = []
            for i in range(3):
                train_img_list.append(
                    compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, i))
            train_img = np.concatenate(train_img_list, axis=1)
            cv2.imwrite(os.path.join("./output", "train_" + str(epoch + 1) + "_step_" + str(step) + ".png"), train_img)
    return mean_iou.avg


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
    model.to(DEVICE)

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