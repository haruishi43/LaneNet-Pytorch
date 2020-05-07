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
        self.resize = Resize((height, width))
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
        ins = np.zeros((no_of_instances, label_instance_img.shape[0], label_instance_img.shape[1]))
        for _ch, label in enumerate(np.unique(label_instance_img)[1:]):
            ins[_ch, label_instance_img == label] = 1

        return ins

    def __len__(self):
        return len(self.data)

    def transform_train(self, img, bin, ins):
        # resize
        img = self.resize(img)
        bin = self.resize(bin)
        ins = self.resize(ins)

        # random horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            bin = TF.hflip(bin)
            ins = TF.hflip(ins)
        
        # make it binary?
        bin = np.zeros([bin.shape[0], bin.shape[1]], dtype=np.uint8)
        mask = np.where((bin[:, :, :] != [0, 0, 0]).all(axis=2))
        bin[mask] = 1

        # split instance gt
        ins = self._split_instance_gt(ins)

        # color jitter
        img = self.color_jitter(img)

        # to Tensor
        img = self.toTensor(img)
        bin = self.toTensor(bin)
        ins = self.toTensor(ins)

        # normalize
        img = self.normalize(img)

        return img, bin, ins

    def transform_test(self, img, bin, ins):
        # resize
        img = self.resize(img)
        bin = self.resize(bin)
        ins = self.resize(ins)

        # make it binary?
        bin = np.zeros([bin.shape[0], bin.shape[1]], dtype=np.uint8)
        mask = np.where((bin[:, :, :] != [0, 0, 0]).all(axis=2))
        bin[mask] = 1

        # split instance gt
        ins = self._split_instance_gt(ins)

        # to Tensor
        img = self.toTensor(img)
        bin = self.toTensor(bin)
        ins = self.toTensor(ins)

        # normalize
        img = self.normalize(img)

        return img, bin, ins

    def __getitem__(self, idx):
        img_path, bin_path, ins_path = self.data[idx]
        img_path = osp.join(self.root, img_path)
        bin_path = osp.join(self.root, bin_path)
        ins_path = osp.join(self.root, ins_path)

        img = np.array(Image.open(img_path))
        bin = np.array(Image.open(bin_path))
        ins = np.array(Image.open(ins_path))

        if self.mode == 'train':
            img, bin, ins = self.transform_train(img, bin, ins)
        elif self.mode == 'test':
            img, bin, ins = self.transform_test(img, bin, ins)

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
            osp.direname(train_dataset_file),
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