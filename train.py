import os
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

import pdb
from config import global_config
from dataset import LaneNetDataset
from model import lanenet_model
from utils import vgg_transform
CFG = global_config.cfg


def init_args():
    '''Args:'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default='/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt', help='Path to text file that has training data information')
    parser.add_argument('--save-path', type=str, default='./model/saved_model/', help='Where to save the trained model')
    parser.add_argument('--pretrained-model', type=str, help='pretrained model path')
    parser.add_argument('--log-path', type=str, default='/home/ubuntu/dev/LaneNet-Pytorch/logs', help='Where to save scalar data from TensorboardX')
    return parser.parse_args()


def create_dataloader(args, split=0.8):
    '''Create pytorch dataloader for training and validation dataset'''
    
    transform = vgg_transform()  # normalize images for VGG
    dataset = LaneNetDataset(args.dataset_file, CFG, transform)
    print('Total Images: ', len(dataset))
    
    # split data into training and validation
    train_size = int(split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=CFG.TRAIN.BATCH_SIZE, 
                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                    batch_size=CFG.TRAIN.VAL_BATCH_SIZE, 
                                    shuffle=False)
    return train_loader, val_loader

if __name__ == '__main__':
    # initialize args
    args = init_args()
    
    # get dataloaders
    train_loader, val_loader = create_dataloader(args)

    # model, optimizers
    model = lanenet_model.LaneNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=CFG.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    
    
    
    # Training
    writer = SummaryWriter(args.log_path)
    
    
    total_step = len(train_loader)
    inc = int(total_step / 16)
    for epoch in range(CFG.TRAIN.EPOCHS):
        scheduler.step()
        
        # Training Data
        model.train()
        for i, (src, binary, instance) in enumerate(train_loader):

            # send to gpu
            src = src.cuda()
            binary = binary.cuda()
            instance = instance.cuda()

            # Forward pass
            total_loss, binary_segmentation_loss, disc_loss, _, _ = model.compute_loss(src, binary, instance)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i+1) % inc == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, CFG.TRAIN.EPOCHS, i+1, total_step, total_loss.cpu().item()))
                niter = epoch*total_step+(i+1)
                writer.add_scalar('Train/Loss', total_loss.cpu().item(), niter)
        
        # Validation Data
        #model.eval()
        #for i, (src, binary, instance) in enumerate(val_loader):
        #    
        #    # send to gpu
        #    src = src.cuda()
        #    binary = binary.cuda()
        #    instance = instance.cuda()

        #    # Forward pass
        #    _, _, _, out_logits, pix_embeddings = model.compute_loss(src, binary, instance)
            
            # calculate accuracy
        
        if epoch % 50 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, '{}/lanenet_{}.pth'.format(args.save_path, epoch))

    
    writer.export_scalars_to_json(f'{args.log_path}/all_scalars.json')
    writer.close()
    print('finished')