#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def compute_loss(net_output, binary_label, instance_label):
    k_binary = 0.7
    k_instance = 0.3
    k_dist = 1.0

    ce_loss_fn = nn.CrossEntropyLoss()
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = ce_loss_fn(binary_seg_logits, binary_label)
    # binary loss OK

    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(0.5, 1.5, 1.0, 1.0, 0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)
    
    binary_loss = binary_loss * k_binary
    instance_loss = var_loss * k_instance  # not OK
    dist_loss = dist_loss * k_dist  # not OK
    total_loss = binary_loss + instance_loss + dist_loss
    
    out = net_output["binary_seg_pred"]
    iou = 0
    batch_size = out.size()[0]
    for i in range(batch_size):
        PR = out[i].squeeze(0).nonzero().size()[0]
        GT = binary_label[i].nonzero().size()[0]
        TP = (out[i].squeeze(0) * binary_label[i]).nonzero().size()[0]
        union = PR + GT - TP
        iou += TP / union
    iou = iou / batch_size
    return total_loss, binary_loss, instance_loss, out, iou


class DiscriminativeLoss(_Loss):
    """
    From https://github.com/nyoki-mtl/pytorch-discriminative-loss/blob/master/src/loss.py
    This is the implementation of following paper:
    https://arxiv.org/pdf/1802.05591.pdf
    This implementation is based on following code:
    https://github.com/Wizaron/instance-segmentation-pytorch
    """
    def __init__(
        self,
        delta_var=0.5,
        delta_dist=1.5,
        norm=2,
        alpha=1.0,
        beta=1.0,
        gamma=0.001,
        usegpu=False,
        size_average=True
    ) -> None:
        super().__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):
        # _assert_no_grad(target)
        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):
        batch_size = embedding.shape[0]
        embed_dim = embedding.shape[1]

        var_loss = torch.tensor(
            0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(
            0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(
            0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H, W)
            seg_gt_b = seg_gt[b]

            labels = torch.unique(seg_gt_b)
            labels = labels[labels != 0]
            num_lanes = len(labels)
            if num_lanes == 0:
                # please refer to issue here:
                # https://github.com/harryhan618/LaneNet/issues/12
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)
                if not seg_mask_i.any():
                    continue
                embedding_i = embedding_b[seg_mask_i]

                mean_i = torch.mean(embedding_i, dim=0)
                centroid_mean.append(mean_i)

                # ---------- var_loss -------------
                var_loss = var_loss + torch.mean(
                    F.relu(
                        torch.norm(embedding_i - mean_i, dim=0) - self.delta_var
                    ) ** 2) / num_lanes
            
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)
                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)  # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist  # diagonal elements are 0, now mask above delta_d

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                            num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size
        return var_loss, dist_loss, reg_loss
