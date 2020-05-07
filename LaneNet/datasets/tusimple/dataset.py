#!/usr/bin/env python3

import os
import os.path as osp
from shutil import copyfile
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from torchvision.transforms import *
import torchvision.transforms.functional as TF


class tuSimpleDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        root='',
        n_labels=5,
        mode='train',
        height=256,
        width=512,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
    ) -> None:
        self.root = root
        self.n_labels = n_labels
        
        dataset_path = osp.join(self.root, dataset_path)
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                info_tmp = line.strip(' ').split()
                d = [a for a in line.rstrip('\n').split(' ')]
                data.append(d)
        self.data = data
        self._shuffle()

        self.mode = mode
        self.resize_img = Resize((height, width), interpolation=Image.BILINEAR)
        self.resize_lbl = Resize((height, width), interpolation=Image.NEAREST)
        self.toTensor = ToTensor()
        self.normalize = Normalize(mean=norm_mean, std=norm_std)
        self.color_jitter = ColorJitter(
            brightness=0.2, contrast=0.15, saturation=0, hue=0
        )

    def _shuffle(self):
        # randomly shuffle all list identically
        random.shuffle(self.data)

    def _split_instance_gt(self, label_instance_img):
        # number of channels, number of unique pixel values, subtracting no label
        # adapted from here https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/dataset.py
        no_of_instances = self.n_labels
        ins = np.zeros(
            (
                label_instance_img.shape[0],
                label_instance_img.shape[1],
                no_of_instances
            ),
            dtype=np.uint8
        )
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
            ins[label_instance_img == label, _ch] = 1
        return ins

    def __len__(self):
        return len(self.data)

    def transform_train(self, img, bin, ins):
        # resize
        img = self.resize_img(img)
        bin = self.resize_lbl(bin)
        ins = self.resize_lbl(ins)

        # random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            bin = TF.hflip(bin)
            ins = TF.hflip(ins)
        
        # make it binary?
        bin = np.array(bin)
        _bin = np.zeros([bin.shape[0], bin.shape[1]], dtype=np.uint8)
        _bin = np.where(bin != 0, 1, 0)

        # split instance gt
        ins = np.array(ins)
        ins = self._split_instance_gt(ins)

        # color jitter
        img = self.color_jitter(img)

        # to Tensor
        img = self.toTensor(img)
        bin = self.toTensor(_bin)
        ins = self.toTensor(ins)

        # normalize
        img = self.normalize(img)

        return img, bin, ins

    def transform_test(self, img, bin, ins):
        # resize
        img = self.resize_img(img)
        bin = self.resize_lbl(bin)
        ins = self.resize_lbl(ins)

        # make it binary?
        bin = np.array(bin)
        _bin = np.zeros([bin.shape[0], bin.shape[1]], dtype=np.uint8)
        _bin = np.where(bin != 0, 1, 0)

        # split instance gt
        ins = np.array(ins)
        ins = self._split_instance_gt(ins)

        # to Tensor
        img = self.toTensor(img)
        bin = self.toTensor(_bin)
        ins = self.toTensor(ins)

        # normalize
        img = self.normalize(img)

        return img, bin, ins

    def __getitem__(self, idx):
        img_path, bin_path, ins_path = self.data[idx]
        img_path = osp.join(self.root, img_path)
        bin_path = osp.join(self.root, bin_path)
        ins_path = osp.join(self.root, ins_path)

        img = Image.open(img_path)
        bin = Image.open(bin_path)
        ins = Image.open(ins_path)

        if self.mode == 'train':
            img, bin, ins = self.transform_train(img, bin, ins)
            bin = bin.squeeze(0)
        elif self.mode == 'test':
            img, bin, ins = self.transform_test(img, bin, ins)
            bin = bin.squeeze(0)
        else:
            img = np.array(img)
            bin = np.array(bin)
            ins = np.array(ins)

        return {
            'image': img,
            'binary': bin,
            'instance': ins,
        }

    @staticmethod
    def create_val(
        train_dataset_file: str,
        val_dataset_file: str,
        val_ratio: float = 0.2, 
    ) -> None:
        new_train_dataset_file = osp.join(
            osp.dirname(train_dataset_file),
            osp.splitext(osp.basename(train_dataset_file))[0] + '_original' + '.txt',
        )
        assert not osp.exists(new_train_dataset_file)
        copyfile(train_dataset_file, new_train_dataset_file)
        data = []
        with open(train_dataset_file, 'r') as f:
            for line in f:
                info_tmp = line.strip(' ').split()
                d = [a for a in line.rstrip('\n').split(' ')]
                data.append(d)

        random.seed(0)  # seed is kept the same
        random.shuffle(data)

        train_data = data[int(len(data)*val_ratio):]
        val_data = data[:int(len(data)*val_ratio)]

        with open(train_dataset_file, 'w') as f:
            for d in train_data:
                info = '{:s} {:s} {:s}\n'.format(
                    d[0], d[1], d[2])
                f.write(info)

        with open(val_dataset_file, 'w') as f:
            for d in val_data:
                info = '{:s} {:s} {:s}\n'.format(
                    d[0], d[1], d[2])
                f.write(info)

        print('saved new dataset')