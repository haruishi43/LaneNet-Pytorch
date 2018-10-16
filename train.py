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
import pdb


from config import global_config
from dataset import LaneNetDataset
from model import lanenet_model
from utils import preprocess_rgb
CFG = global_config.cfg


def init_args():
    '''
    Args:
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default='/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt', help='Path to text file that has training data information')
    parser.add_argument('--net', type=str, default='vgg', help='choose which base network to use')
    parser.add_argument('--weights-path', type=str, help='pretrained weights path')
    return parser.parse_args()



if __name__ == '__main__':
    # initialize args
    args = init_args()
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    dataset = LaneNetDataset(args.dataset_file, CFG, transform)
    
    print(len(dataset))
    # split data into training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=CFG.TRAIN.BATCH_SIZE, 
                                           shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                    batch_size=CFG.TRAIN.VAL_BATCH_SIZE, 
                                    shuffle=False)
    
    model = lanenet_model.LaneNet().cuda()
    optimizer = optim.Adam(model.parameters(), lr=CFG.TRAIN.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
    
    
    
    total_step = len(train_loader)
    inc = int(total_step / 16)
    for epoch in range(CFG.TRAIN.EPOCHS):
        model.train()
        scheduler.step()
        
        for i, (src, binary, instance) in enumerate(train_loader):

            # dataloader automatically converts to tensor
    #         print(src.shape)
    #         print(binary.shape)
    #         print(instance.shape)

            # send to gpu
            src = src.cuda()
            binary = binary.cuda()
            instance = instance.cuda()

            # Forward pass
            total_loss, binary_segmentation_loss, pix_embedding, disc_loss = model.compute_loss(src, binary, instance)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (i+1) % inc == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, CFG.TRAIN.EPOCHS, i+1, total_step, total_loss.cpu().item()))
    
        if epoch % 200 == 0:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    }, './model/checkpoints/lanenet_{}.pth'.format(epoch))

    print('finished')
