import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

if __name__ == '__main__':
    import vgg_encoder
    import fcn_decoder
else:
    from model import vgg_encoder
    from model import fcn_decoder


class LaneNet(nn.Module):

    def __init__(self, use_cuda=True):
        super(LaneNet, self).__init__()
        self.encoder = vgg_encoder.VGGEncoder()
        self.decoder = fcn_decoder.FCNDecoder()
        self.conv1 = nn.Conv2d(64, 3, kernel_size=1, bias=False)  # pixembedding
        self.entropy = nn.CrossEntropyLoss()
        self.use_cuda = use_cuda
    
    def forward(self, src):
        # encode
        ret = self.encoder(src)
        # decode
        decode_logits, decode_deconv  = self.decoder(ret)
        pix_embedding = F.relu(self.conv1(decode_deconv))
        return (decode_logits, pix_embedding)

    def inference(self, src):
        decode_logits, pix_embedding  = self.forward(src)
        
        binary_seg_ret = F.softmax(decode_logits)
        binary_seg_ret = np.argmax(binary_seg_ret, dim=1)
        
        return (binary_seg_ret, pix_embedding) 
    
    def compute_loss(self, src, binary, instance):
        decode_logits, pix_embedding = self.forward(src)

        # step 1:
        # calculate loss between binary and decode logits
        #
        # use softmax_cross_entropy
        decode_logits_reshape = decode_logits.view([decode_logits.shape[0], 
                                                    decode_logits.shape[1], 
                                                    decode_logits.shape[2] * decode_logits.shape[3]])
        binary_reshape = binary.view(binary.shape[0],
                                     binary.shape[1]*binary.shape[2])
        binary_reshape = torch.div(binary_reshape, 255)
        binary_reshape = binary_reshape.long()

        binary_segmentation_loss = self.entropy(decode_logits_reshape, binary_reshape)

        # step 2:
        # calculate discriminative loss between deconv and instance
        disc_loss, l_var, l_dist, l_reg = \
                self.discriminative_loss(pix_embedding, instance, 0.5, 1.5, 1.0, 1.0, 0.001)
        
        total_loss = 0.7*binary_segmentation_loss + 0.3*disc_loss
        
        return total_loss, binary_segmentation_loss, pix_embedding, disc_loss
        
    def discriminative_loss(self, prediction, correct_label,
                        delta_v, delta_d, param_var, param_dist, param_reg):
        
        # saving list (maybe implement dynamic tensor?)
        output_ta_loss = []
        output_ta_var = []
        output_ta_dist = []
        output_ta_reg = []
        
        # for each batch calculate the loss
        i = 0
        while i < prediction.shape[0]:
            # calculate discrimitive loss for single image
            single_prediction = prediction[i]
            single_label = correct_label[i]
            # pdb.set_trace()
            disc_loss, l_var, l_dist, l_reg = self.discriminative_loss_single(
                single_prediction, single_label, delta_v, delta_d, param_var, param_dist, param_reg)
            
            output_ta_loss.append(disc_loss.unsqueeze(0))
            output_ta_var.append(l_var.unsqueeze(0))
            output_ta_dist.append(l_dist.unsqueeze(0))
            output_ta_reg.append(l_reg.unsqueeze(0))
            
            i += 1  # next image in batch
        
        out_loss_op = torch.cat(output_ta_loss)
        out_var_op = torch.cat(output_ta_var)
        out_dist_op = torch.cat(output_ta_dist)
        out_reg_op = torch.cat(output_ta_reg)
        
        # calculate mean of the batch
        disc_loss = out_loss_op.mean()
        l_var = out_var_op.mean()
        l_dist = out_dist_op.mean()
        l_reg = out_reg_op.mean()

        return disc_loss, l_var, l_dist, l_reg
        
    def discriminative_loss_single(
            self,
            prediction,
            correct_label,
            delta_v,
            delta_d,
            param_var,
            param_dist,
            param_reg):
        '''
        The example partition loss function mentioned in the paper equ(1)
        :param prediction: inference of network
        :param correct_label: instance label
        :param delta_v: cutoff variance distance
        :param delta_d: curoff cluster distance
        :param param_var: weight for intra cluster variance
        :param param_dist: weight for inter cluster distances
        :param param_reg: weight regularization
        '''
        
        feature_dim = prediction.shape[0]
        # Make it a single line
        correct_label = correct_label.view([correct_label.shape[0] * correct_label.shape[1]]).float()
        reshaped_pred = prediction.view([feature_dim, prediction.shape[1] * prediction.shape[2]]).float()
        
        # Get unique labels
        unique_labels, unique_id = torch.unique(correct_label, sorted=True, return_inverse=True)
        ids, counts = np.unique(unique_id, return_counts=True)
        num_instances = len(counts)
        counts = torch.tensor(counts, dtype=torch.float32)
        if self.use_cuda:
            counts = counts.cuda()
        
        # Calculate the pixel embedding mean vector
        if self.use_cuda:
            segmented_sum = torch.zeros(feature_dim, num_instances).cuda().scatter_add(1, unique_id.repeat([feature_dim,1]), reshaped_pred)
        else:
            segmented_sum = torch.zeros(feature_dim, num_instances).scatter_add(1, unique_id.repeat([feature_dim,1]), reshaped_pred)
        
        mu = torch.div(segmented_sum, counts)
        mu_expand = torch.gather(mu, 1, unique_id.repeat([feature_dim,1]))

        # Calculate loss(var)
        distance = (mu_expand - reshaped_pred).t().norm(dim=1)
        distance = distance - delta_v
        distance = torch.clamp(distance, min=0.)   # min is 0.
        distance = distance.pow(2)
        
        if self.use_cuda:
            l_var = torch.zeros(num_instances).cuda().scatter_add(0, unique_id, distance)
        else:
            l_var = torch.zeros(num_instances).scatter_add(0, unique_id, distance_3)
        l_var = torch.div(l_var, counts)
        l_var = l_var.sum()
        l_var = torch.div(l_var, num_instances)  # single value 
   
        # Calculate the loss(dist) of the formula
        mu_diff = []
        for i in range(feature_dim):
            for j in range(feature_dim):
                if i != j:
                    diff = mu[i] - mu[j]
                    mu_diff.append(diff.unsqueeze(0))
                    
        mu_diff = torch.cat(mu_diff)
        
        mu_norm = mu_diff.norm(dim=1)
        mu_norm = (2. * delta_d - mu_norm)
        mu_norm = torch.clamp(mu_norm, min=0.)
        mu_norm = mu_norm.pow(2)
        
        l_dist = mu_norm.mean()
        
        # Calculate the regular term loss mentioned in the original Discriminative Loss paper
        l_reg = mu.norm(dim=1).mean()

        # Consolidation losses are combined according to the parameters mentioned in the original Discriminative Loss paper
        param_scale = 1.
        l_var = param_var * l_var
        l_dist = param_dist * l_dist
        l_reg = param_reg * l_reg

        loss = param_scale * (l_var + l_dist + l_reg)

        return loss, l_var, l_dist, l_reg

    
if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0,'.')
    from config import global_config
    from dataset import LaneNetDataset
    from utils import preprocess_rgb

    TRAIN_FILE = '/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt'
    CFG = global_config.cfg
    
    dataset = LaneNetDataset(TRAIN_FILE, CFG)
    inputs = next(iter(dataset))  # (src, binary, instance)
    
    lane_net = LaneNet().cuda()
    
    src = preprocess_rgb(inputs[0])
    print(src)
    binary = torch.tensor(inputs[1]).unsqueeze(0).cuda()
    inference = torch.tensor(inputs[2]).unsqueeze(0).cuda()
    
    total_loss, binary_segmentation_loss, pix_embedding, disc_loss = lane_net.compute_loss(src, binary, inference)
    print(total_loss)