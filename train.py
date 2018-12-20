import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import pdb

from arguments import init_args
from config import global_config
from dataset import LaneNetDataset
from model import lanenet_model
CFG = global_config.cfg




def create_dataloader(args, split=0.8):
    '''Create pytorch dataloader for training and validation dataset'''
    
    dataset = LaneNetDataset(args.dataset_file, CFG, True)
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


def calculate_binary_accuracy(logits, label):
    '''Calculate binary segmentation accuracy'''
    # input logits are separated in two class
    # check which class scored highest
    binary_prob = F.softmax(logits, dim=1)
    out = binary_prob.argmax(1)  # which class scored the highest?

    # Get indices where label
    lane_indices = (binary == 1).nonzero()  # white pixels are 1
    
    # To numpy for cpu calculation
    out = out.cpu().numpy()
    lane_indices = lane_indices.cpu().numpy()
    masked_out = np.take(out, lane_indices)

    correct = np.count_nonzero(masked_out)
    accuracy = correct / len(masked_out)
    
    return accuracy


def train(args):
    
    # get dataloaders
    train_loader, val_loader = create_dataloader(args)
    
    # model, optimizers
    model = lanenet_model.LaneNet(device=args.main_device).to(args.main_device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    
    #TODO: Load resume path
    
    total_step = len(train_loader)
    inc = int(total_step / 16)
    for epoch in range(CFG.TRAIN.EPOCHS):
        scheduler.step()
        
        # Training Data
        model.train()
        for i, (src, binary, instance) in enumerate(train_loader):

            # send to gpu
            src = src.to(args.main_device)
            binary = binary.to(args.main_device)
            instance = instance.to(args.main_device)

            # Forward pass
            total_loss, binary_seg_loss, disc_loss, _, _ = model.compute_loss(src, binary, instance)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i+1) % inc == 0:
                total_loss = total_loss.cpu().item()
                binary_seg_loss = binary_seg_loss.cpu().item()
                disc_loss = disc_loss.cpu().item()
                print ('Epoch [{}/{}], Step [{}/{}], Total Loss: {:.4f}, Binary Loss: {:.4f}, Disc Loss: {:.4f}' 
                   .format(epoch+1, CFG.TRAIN.EPOCHS, i+1, total_step, total_loss, binary_seg_loss, disc_loss))
                niter = epoch*total_step+(i+1)
        
        # Validation Data
        model.eval()
        with torch.no_grad():
            for i, (src, binary, instance) in enumerate(val_loader):

                # send to gpu
                src = src.to(args.main_device)
                binary = binary.to(args.main_device)
                instance = instance.to(args.main_device)

                # Forward pass

                _, _, _, binary_logits, pix_embeddings = model.compute_loss(src, binary, instance)

                # Calculate accuracy
                if (i+1)%inc == 0:
                    accuracy = calculate_binary_accuracy(binary_logits, binary)
                    print(accuracy)



            if (epoch+1) % 50 == 0:
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        }, '{}/lanenet_{}.pth'.format(args.save_path, epoch))

    print('finished')