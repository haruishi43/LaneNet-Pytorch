#!/usr/bin/env python3

import sys
import time
import datetime
import os.path as osp

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..losses import compute_loss
from ..utils import (
    AverageMeter, compose_img, mkdir_if_missing,
    save_checkpoint
)


class Engine(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer = None,
        lr_scheduler: object = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        use_gpu: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.use_gpu = use_gpu
        self.writer = None

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=5,
        start_eval=0,
        eval_freq=10,
    ) -> None:
        if self.writer is None:
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
                self.eval(
                    epoch,
                    self.writer,
                    save_dir=save_dir,
                )
                self._save_checkpoint(epoch, save_dir)
        
        if max_epoch > 0:
            print('=> Final test')
            self.eval(
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

        self.model.train()
        num_batches = len(self.train_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            
            image, binary, instance = self._parse_data(data)
            if self.use_gpu:
                image = image.cuda()
                binary = binary.cuda()
                instance = instance.cuda()

            # forward pass
            net_output = self.model(image)

            # compute loss
            (
                total_loss, binary_loss,
                instance_loss, out, train_iou
            ) = compute_loss(
                net_output, binary, instance
            )

            # update loss in AverageMeter instance
            total_losses.update(total_loss.item(), image.size()[0])
            binary_losses.update(binary_loss.item(), image.size()[0])
            instance_losses.update(instance_loss.item(), image.size()[0])
            mean_iou.update(train_iou, image.size()[0])

            # reset gradients
            self.optimizer.zero_grad()

            # backpropagate
            total_loss.backward()

            # update weights
            self.optimizer.step()

            # update batch time
            batch_time.update(time.time() - end)
            end = time.time()

            if (batch_idx+1) % print_freq == 0:
                # estimate remaining time
                eta_seconds = batch_time.avg * (
                    num_batches - (batch_idx+1) + (
                        max_epoch - (epoch+1)
                    ) * num_batches
                )
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    'Epoch: [{0}/{1}][{2}/{3}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
                    'Tot Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'
                    'Bin Loss {binary_loss.val:.4f} ({binary_loss.avg:.4f})\n'
                    'Ins Loss {instance_loss.val:.4f} ({instance_loss.avg:.4f})\t'
                    'Mean IoU {mean_iou.val:.2f} ({mean_iou.avg:.2f})\n'
                    'Lr {lr:.6f}\t'
                    'eta {eta}'.format(
                        epoch + 1,
                        max_epoch,
                        batch_idx + 1,
                        num_batches,
                        batch_time=batch_time,
                        total_loss=total_losses,
                        binary_loss=binary_losses,
                        instance_loss=instance_losses,
                        mean_iou=mean_iou,
                        lr=self.optimizer.param_groups[0]['lr'],
                        eta=eta_str
                    )
                )

            if writer is not None:
                n_iter = epoch*num_batches + batch_idx
                writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                writer.add_scalar('Train/Total Loss', total_losses.avg, n_iter)
                writer.add_scalar('Train/Binary Loss', binary_losses.avg, n_iter)
                writer.add_scalar('Train/Instance Loss', instance_losses.avg, n_iter)
                writer.add_scalar('Train/Acc', mean_iou.avg, n_iter)
                writer.add_scalar(
                    'Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter
                )
            
            if self.scheduler is not None:
                self.scheduler.step()

    @torch.no_grad()
    def eval(
        self,
        epoch,
        writer,
    ) -> None:
        self.model.eval()
        
        batch_time = AverageMeter()
        total_losses = AverageMeter()
        binary_losses = AverageMeter()
        instance_losses = AverageMeter()
        mean_iou = AverageMeter()
        
        val_img_list = []
        num_batches = len(self.val_loader)
        end = time.time()
        for batch_idx, data in enumerate(self.val_loader):
            image, binary, instance = self._parse_data(data)
            if self.use_gpu:
                image = image.cuda()
                binary = binary.cuda()
                instance = instance.cuda()

            # output process
            net_output = model(image)
            (
                total_loss, binary_loss, instance_loss,
                out, val_iou,
            ) = compute_loss(
                net_output, binary, instance
            )
            total_losses.update(total_loss.item(), image.size()[0])
            binary_losses.update(binary_loss.item(), image.size()[0])
            instance_losses.update(instance_loss.item(), image.size()[0])
            mean_iou.update(val_iou, image.size()[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx+1 == num_batches:
                img_list = []
                for i in range(3):
                    img_list.append(
                        compose_img(
                            image, out, binary,
                            net_output["instance_seg_logits"],
                            instance, i
                        )
                    )
                img = np.concatenate(img_list, axis=1)
                mkdir_if_missing("./output")
                cv2.imwrite(osp.join(f"./output/val_{str(epoch+1)}.png"), img)

        print(
            "Epoch {ep} Validation Report | ETA: {et:.2f}|Total:{tot:.5f}|Binary:{bin:.5f}|Instance:{ins:.5f}|IoU:{iou:.5f}".format(
                ep=epoch + 1,
                et=batch_time.val,
                tot=total_losses.avg,
                bin=binary_losses.avg,
                ins=instance_losses.avg,
                iou=mean_iou.avg,
            )
        )

    @torch.no_grad()
    def test(
        self,
        test_loader: DataLoader,
    ):
        self.model.eval()
        
        batch_time = AverageMeter()
        total_losses = AverageMeter()
        binary_losses = AverageMeter()
        instance_losses = AverageMeter()
        mean_iou = AverageMeter()
        
        num_batches = len(test_loader)
        end = time.time()
        for batch_idx, data in enumerate(test_loader):
            image, binary, instance = self._parse_data(data)
            if self.use_gpu:
                image = image.cuda()
                binary = binary.cuda()
                instance = instance.cuda()

            # output process
            net_output = model(image)
            (
                total_loss, binary_loss, instance_loss, out, iou
            ) = compute_loss(
                net_output, binary, instance
            )
            total_losses.update(total_loss.item(), image.size()[0])
            binary_losses.update(binary_loss.item(), image.size()[0])
            instance_losses.update(instance_loss.item(), image.size()[0])
            mean_iou.update(iou, image.size()[0])
            if batch_idx+1 == num_batches:
                img_list = []
                for i in range(3):
                    img_list.append(
                        compose_img(
                            image, out, binary,
                            net_output["instance_seg_logits"],
                            instance, i
                        )
                    )
                img = np.concatenate(img_list, axis=1)
                mkdir_if_missing('./output')
                cv2.imwrite(osp.join("./output/test.png"), img)

    def _parse_data(self, data):
        image = data['image']
        binary = data['binary']
        instance = data['instance']
        return image, binary, instance

    def _save_checkpoint(self, epoch, save_dir, is_best=False):
        save_checkpoint(
            {
                'state_dict': self.model.state_dict(),
                'epoch': epoch + 1,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
            },
            save_dir,
            is_best=is_best
        )