import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class FCNDecoder(nn.Module):
    
    def __init__(self, input_channel=512):
        super(FCNDecoder, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channel, 64, kernel_size=1, bias=False)
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False)
        self.conv2 = nn.Conv2d(input_channel, 64, kernel_size=1, bias=False)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, bias=False)
        self.conv3 = nn.Conv2d(input_channel//2, 64, kernel_size=1, bias=False)
        self.deconv3 = nn.ConvTranspose2d(64, 64, kernel_size=8, stride=8, bias=False)
        self.conv4 = nn.Conv2d(64, 2, kernel_size=1, bias=False)
        
        self.train()
        
    def forward(self, tensor_dict):
        
        x = tensor_dict['pool5']
        score = self.conv1(x)
        
        deconv = self.deconv1(score)
        x = tensor_dict['pool4']
        score = self.conv2(x)
        
        score = torch.add(deconv, score)
        
        deconv = self.deconv2(score)
        x = tensor_dict['pool3']
        score = self.conv3(x)
        
        score = torch.add(deconv, score)
        
        deconv_final = self.deconv3(score)
        score_final = self.conv4(deconv_final)
        
        return (score_final, deconv_final)

if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0,'..')
    from config import global_config
    from dataset import LaneNetDataset
    from model import vgg_encoder

    TRAIN_FILE = '/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt'
    CFG = global_config.cfg

    dataset = LaneNetDataset(TRAIN_FILE, CFG)
    inputs = next(iter(dataset))  # (src, binary, instance)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    # preprocess images
    src_tensor = preprocess(inputs[0])
    src_tensor = src_tensor.unsqueeze(0).cuda()

    vgg = vgg_encoder.VGGEncoder().cuda()
    fcn = FCNDecoder().cuda()

    # test:
    output = vgg(src_tensor)
    score_final, deconv_final = fcn.forward(output)

    print('binary shape: ', score_final.shape)  # binary segmentation
    print('embedding shape: ', deconv_final.shape)  # embedding

    pass
