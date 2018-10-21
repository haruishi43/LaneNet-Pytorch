import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from config import global_config
from dataset import LaneNetDataset
from model.lanenet_model import LaneNet

training_file = '/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt'
CFG = global_config.cfg

if __name__ == '__main__':
    
    import torchvision.transforms.functional as VF
    
    dataset = LaneNetDataset(training_file, CFG, True)
    
    # try for one
    inputs = next(iter(dataset))  # (src, binary, instance)
    
    
    checkpoint = torch.load('../model/saved_model/lanenet_150.pth')
    lane_net.load_state_dict(checkpoint['model_state_dict'])
    
    
    src = inputs[0].unsqueeze(0).cuda()
    
    lane_net.cuda()
    lane_net.eval()
    
    binary_seg_ret, pix_embedding = lane_net.inference(src)
    
    binary_seg = binary_seg_ret.cpu()
    binary = VF.to_pil_image(binary_seg.int())
    
    # save image.