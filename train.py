import os
import argparse
import numpy as np

from dataset import LaneNetDataset
from config import global_config
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

    dataset = LaneNetDataset(args.dataset_file, CFG)
    
    print(len(dataset))


    pass
