import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms


if __name__ == '__main__':
    import vgg_encoder
    import fcn_decoder
    import discriminative_loss
else:
    from model import vgg_encoder
    from model import fcn_decoder
    from model import discriminative_loss


class LaneNet(nn.Module):

    def __init__(self, device='cpu'):
        super(LaneNet, self).__init__()
        self.encoder = vgg_encoder.VGGEncoder()
        self.decoder = fcn_decoder.FCNDecoder()
        self.conv1 = nn.Conv2d(64, 3, kernel_size=1, bias=False)  # pixembedding
        self.entropy = nn.CrossEntropyLoss()
        self.discriminative_loss = discriminative_loss.DiscriminativeLoss(device=device)
    
    def forward(self, src):
        # encode
        ret = self.encoder(src)
        # decode
        decode_logits, decode_deconv  = self.decoder(ret)
        pix_embedding = F.relu(self.conv1(decode_deconv))
        return (decode_logits, pix_embedding)

    def inference(self, src):
        decode_logits, pix_embedding  = self.forward(src)
        
        binary_seg_ret = F.softmax(decode_logits, dim=0)
        binary_seg_ret = binary_seg_ret.argmax(1)
        
        return (binary_seg_ret, pix_embedding) 
    
    def compute_loss(self, src, binary, instance):
        # preprocess label images
        if len(binary.shape) == 4:
            binary = binary.squeeze(1)
        if len(instance.shape) == 4:
            instance = instance.squeeze(1)
            
        # Get predictions
        decode_logits, pix_embedding = self.forward(src)

        # step 1:
        # calculate loss between binary and decode logits
        # use softmax_cross_entropy
        decode_logits_reshape = decode_logits.view([decode_logits.shape[0], 
                                                    decode_logits.shape[1], 
                                                    decode_logits.shape[2] * decode_logits.shape[3]])
        binary_reshape = binary.view(binary.shape[0],
                                     binary.shape[1]*binary.shape[2])
        binary_reshape = binary_reshape.long()
        binary_segmentation_loss = self.entropy(decode_logits_reshape, binary_reshape)

        # step 2:
        # calculate discriminative loss between deconv and instance
        disc_loss, l_var, l_dist, l_reg = \
                self.discriminative_loss(pix_embedding, instance)
        
        total_loss = 0.7*binary_segmentation_loss + 0.3*disc_loss
        
        return total_loss, binary_segmentation_loss, disc_loss, decode_logits, pix_embedding
        
    
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0,'.')
    from config import global_config
    from dataset import LaneNetDataset

    TRAIN_FILE = '/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt'
    CFG = global_config.cfg
    
    dataset = LaneNetDataset(TRAIN_FILE, CFG, True)
    inputs = next(iter(dataset))  # (src, binary, instance)
    
    lane_net = LaneNet().cuda()
    
    src = inputs[0].unsqueeze(0).cuda()
    print(src)
    binary = inputs[1].unsqueeze(0).cuda()
    inference = inputs[2].unsqueeze(0).cuda()
    
    total_loss, _, _, _, _ = lane_net.compute_loss(src, binary, inference)
    print(total_loss)