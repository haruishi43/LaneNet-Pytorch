#!/usr/bin/env python3

import sys
import time
import datetime
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..losses import compute_loss
from ..utils import AverageMeter, compose_img


class Engine(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: object,
        train_loader: DataLoader,
        val_loader: DataLoader,
        use_gpu: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_gpu = use_gpu
        self.writer = None

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        start_eval=0,
        eval_freq=5,
        test_only=False,
    ) -> None:
        if test_only:
            self.test(
                0
            )
            return

        with self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)

        time_start = time.time()
        print('=> Start training')
        
        for epoch in range(0, max_epoch):
            print(f"Epoch: {epoch}")
            self.train(epoch, max_epoch, self.writer, print_freq)

            if (epoch + 1) >= start_eval \
                and eval_freq > 0 \
                and (epoch+1) % eval_freq == 0 \
                and (epoch + 1) != max_epoch:
                self.test(
                    epoch,
                    self.writer,
                    save_dir=save_dir,
                )
                self._save_checkpoint(epoch, save_dir)
        
        if max_epoch > 0:
            print('=> Final test')
            self.test(
                epoch,
                self.writer,
                save_dir=save_dir,
            )
            self._save_checkpoint(epoch, save_dir)

        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()

    def train(
        self,
        epoch,
        max_epoch,
        writer,
        print_freq=10,
    ) -> None:
        batch_time = AverageMeter()
        mean_iou = AverageMeter()
        total_losses = AverageMeter()
        binary_losses = AverageMeter()
        instance_losses = AverageMeter()
        end = time.time()
        step = 0

        for idx, data in enumerate(iter(self.train_loader)):
            step += 1
            
            image, binary, instance = self._parse_data(data)
            if self.use_gpu:
                image = image.cuda()
                binary = binary.cuda()
                instance = instance.cuda()

            # forward pass
            net_output = model(image)

            # compute loss
            total_loss, binary_loss, instance_loss, out, train_iou = compute_loss(
                net_output, binary, instance
            )

            # update loss in AverageMeter instance
            total_losses.update(total_loss.item(), image.size()[0])
            binary_losses.update(binary_loss.item(), image.size()[0])
            instance_losses.update(instance_loss.item(), image.size()[0])
            mean_iou.update(train_iou, image.size()[0])

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
                        batch=idx + 1,
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
                        compose_img(image, out, binary, net_output["instance_seg_logits"], instance, i))
                train_img = np.concatenate(train_img_list, axis=1)
                cv2.imwrite(osp.join("./output", "train_" + str(epoch + 1) + "_step_" + str(step) + ".png"), train_img)

    @torch.no_grad()
    def test(
        self,
        epoch,
        writer,
    ) -> None:
        self.model.eval()
        step = 0
        batch_time = AverageMeter()
        total_losses = AverageMeter()
        binary_losses = AverageMeter()
        instance_losses = AverageMeter()
        mean_iou = AverageMeter()
        end = time.time()
        val_img_list = []
        # val_img_md5 = open(os.path.join(im_path, "val_" + str(epoch + 1) + ".txt"), "w")
        for idx, data in enumerate(val_loader):
            step += 1
            image, binary, instance = self._parse_data(data)
            if self.use_gpu:
                image = image.cuda()
                binary = binary.cuda()
                instance = instance.cuda()

            # output process
            net_output = model(image)
            total_loss, binary_loss, instance_loss, out, val_iou = compute_loss(
                net_output, binary, instance)
            total_losses.update(total_loss.item(), image.size()[0])
            binary_losses.update(binary_loss.item(), image.size()[0])
            instance_losses.update(instance_loss.item(), image.size()[0])
            mean_iou.update(val_iou, image.size()[0])

            # if step % 100 == 0:
            #    val_img_list.append(
            #        compose_img(image_data, out, binary_label, net_output["instance_seg_logits"], instance_label, 0))
            #    val_img_md5.write(input_data["img_name"][0] + "\n")
            #    lane_cluster_and_draw(image_data, net_output["binary_seg_pred"], net_output["instance_seg_logits"], input_data["o_size"], input_data["img_name"], json_path)
        batch_time.update(time.time() - end)
        end = time.time()

        print(
            "Epoch {ep} Validation Report | ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
                ep=epoch + 1,
                et=batch_time.val,
                tot=total_losses.avg,
                bin=binary_losses.avg,
                ins=instance_losses.avg,
                iou=mean_iou.avg,
            ))
        sys.stdout.flush()
        val_img = np.concatenate(val_img_list, axis=1)
        # cv2.imwrite(os.path.join(im_path, "val_" + str(epoch + 1) + ".png"), val_img)
        # val_img_md5.close()

    def _parse_data(self, data):
        image = data['image']
        binary = data['binary']
        instance = data['instance']
        return image, binary, instance