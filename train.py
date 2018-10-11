import os
import argparse
import numpy as np

from config import global_config
CFG = global_config.cfg


def init_args():
    '''
    Args:
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str, default='./data/training_data/', help='Data for training')
    parser.add_argument('--net', type=str, default='vgg', help='choose which base network to use')
    parser.add_argument('--weights-path', type=str, help='pretrained weights path')

    return parser.parse_args()



if __name__ == '__main__':
    # initialize args
    args = init_args()

    print('hello')


    pass
