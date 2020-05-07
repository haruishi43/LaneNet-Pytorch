#!/usr/bin/env python3

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from ..losses import compute_loss
from ..utils import AverageMeter


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

    def run(
        self,
        save_dir='log',
        max_epoch=0,
        start_epoch=0,
        print_freq=10,
        test_only=False,
    ) -> None:
        if test_only:
            self.test(
                0
            )

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

        for batch_idx, batch in enumerate(iter(self.train_loader)):
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

    @torch.no_grad()
    def test(
        self,
        epoch,
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
        for batch_idx, input_data in enumerate(val_loader):
            step += 1
            image_data = input_data["input_tensor"]
            instance_label = input_data["instance_label"]
            binary_label = input_data["binary_label"]

            # output process
            net_output = model(image_data)
            total_loss, binary_loss, instance_loss, out, val_iou = compute_loss(net_output, binary_label, instance_label)
            total_losses.update(total_loss.item(), image_data.size()[0])
            binary_losses.update(binary_loss.item(), image_data.size()[0])
            instance_losses.update(instance_loss.item(), image_data.size()[0])
            mean_iou.update(val_iou, image_data.size()[0])

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