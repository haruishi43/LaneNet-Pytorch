#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VGGEncoder
from .decoders import FCNDecoder


class LaneNet(nn.Module):
    def __init__(self, arch="VGG"):
        super().__init__()
        # no of instances for segmentation
        self.no_of_instances = 5
        encode_num_blocks = 5
        in_channels = [3, 64, 128, 256, 512]
        out_channels = in_channels[1:] + [512]
        self._arch = arch
        if self._arch == 'VGG':
            self._encoder = VGGEncoder(
                encode_num_blocks,
                in_channels,
                out_channels
            )

            decode_layers = ["pool5", "pool4", "pool3"]
            decode_channels = out_channels[:-len(decode_layers) - 1:-1]
            decode_last_stride = 8
            self._decoder = FCNDecoder(
                decode_layers,
                decode_channels,
                decode_last_stride
            )
        elif self._arch == 'ESPNet':
            raise NotImplementedError
        elif self._arch == 'ENNet':
            raise NotImplementedError

        self._pix_layer = nn.Conv2d(
            64, self.no_of_instances,
            kernel_size=1, bias=False
        )
        self.relu = nn.ReLU()

    def forward(self, input_tensor):
        encode_ret = self._encoder(input_tensor)
        decode_ret = self._decoder(encode_ret)

        decode_logits = decode_ret['logits']
        binary_seg_ret = torch.argmax(F.softmax(decode_logits, dim=1), dim=1, keepdim=True)
        decode_deconv = decode_ret['deconv']
        pix_embedding = self.relu(self._pix_layer(decode_deconv))

        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': decode_logits,
        }
