import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms

from model import vgg_encoder
from model import fcn_decoder


class LaneNet:

    def __init__(self):
        self.encoder = vgg_encoder.VGGEncoder()
        self.decoder = fcn_decoder.FCNDecoder()
        self.conv1 = nn.Conv2d(64, 3, kernel_size=1, bias=False)  # pixembedding

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    def inference(self, src):
        decode_logits, decode_deconv  = self.run_model(src)
        
        binary_seg_ret = F.softmax(decode_logits)
        binary_seg_ret = np.argmax(binary_seg_ret, dim=1)
        
        pix_embedding = F.relu(self.conv1(decode_deconv))
        return (binary_seg_ret, pix_embedding)
    
    def run_model(self, src):
        src_tensor = self.preprocess(src)

        # have to check if batch or not
        if len(src_tensor) != 4:
            src_tensor = src_tensor.unsqueeze(0)
        
        # encode
        ret = self.encoder(src_tensor)
        # decode
        decode_logits, decode_deconv  = self.decoder(ret)
        return (decode_logits, decode_deconv)

    def compute_loss(self, src, binary, instance):

        decode_logits, decode_deconv  = self.run_model(src)

        # step 1:
        # calculate loss between binary and decode logits
        #
        # use softmax_cross_entropy
        binary_segmenatation_loss = torch.sum(- binary * F.log_softmax(decode_logits, -1), -1)
        binary_segmenatation_loss = binary_segmenatation_loss.mean()

        # step 2:
        # calculate discrimitive loss between deconv and instance
        # change deconv into pix_embedding

        # then calculate discrimitive loss
        pix_embedding = F.relu(self.conv1(decode_deconv))
        disc_loss, l_var, l_dist, l_reg = \
                lanenet_discriminative_loss.discriminative_loss(
                    pix_embedding, instance, 3, 0.5, 1.5, 1.0, 1.0, 0.001)
        
        total_loss = 0.7*binary_segmentation_loss + 0.3*disc_loss
        
        ret = {
            'total_loss': total_loss,
            'binary_seg_logits': decode_logits,
            'instance_seg_logits': pix_embedding,
            'binary_seg_loss': binary_segmenatation_loss,
            'discriminative_loss': disc_loss
        }
        
        return ret
        
    def discrimitive_loss(self, prediction, correct_label, feature_dim,
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
            single_label = correct_lable[i]
            # pdb.set_trace()
            disc_loss, l_var, l_dist, l_reg = single_discrimitive_loss(
                single_prediction, single_label, feature_dim, delta_v, delta_d, param_var, param_dist, param_reg)
            
            output_ta_loss.append(disc_loss.unsqueeze(0))
            output_ta_va.append(l_var.unsqueeze(0))
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
        l_dist = out_vdist_op.mean()
        l_reg = out_reg_op.mean()

        return disc_loss, l_var, l_dist, l_reg
        
    def discriminative_loss_single(
            prediction,
            correct_label,
            feature_dim,
            delta_v,
            delta_d,
            param_var,
            param_dist,
            param_reg):
        """
        The example partition loss function mentioned in the paper equ(1)
        :param prediction: inference of network
        :param correct_label: instance label
        :param feature_dim: feature dimension of prediction
        :param delta_v: cutoff variance distance
        :param delta_d: curoff cluster distance
        :param param_var: weight for intra cluster variance
        :param param_dist: weight for inter cluster distances
        :param param_reg: weight regularization
        """

        # Make it a single line
        correct_label = correct_label.view([correct_label.shape[0] * correct_label.shape[1]]).float()
        reshaped_pred = prediction.view([feature_dim, prediction[0] * prediction[1]]).float()
        
        # Get unique labels
        unique_labels, unique_id = torch.unique(correct_label, sorted=True, return_inverse=True)
        ids, counts = np.unique(unique_id, return_counts=True)
        num_instances = len(counts)
        counts = torch.tensor(counts, dtype=torch.float32)
        
        # Calculate the pixel embedding mean vector
        segmented_sum = torch.zeros(feature_dim, num_instances).scatter_add(1, unique_id.repeat([feature_dim,1]), reshaped_pred)
        mu = torch.div(segmented_sum, counts)
        mu_expand = torch.gather(mu, 1, unique_id.repeat([feature_dim,1]))

        # Calculate loss(var)
        distance = (mu_expand - reshaped_pred).t().norm(dim=1)
        distance -= torch.tensor(delta_v, dtype=torch.float32)
        distance = torch.clamp(distance, min=0.)   # min is 0.
        distance = distance.pow(2)
        
        l_var = torch.zeros(num_instances).scatter_add(0, unique_id, distance)
        l_var = torch.div(l_var, counts)
        l_var = l_var.sum()
        l_var = torch.div(l_var, num_instances)  # single value 
   
        # Calculate the loss(dist) of the formula
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
    
    pass