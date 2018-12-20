import argparse

def init_args():
    '''Args:'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', type=str, default='/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt', help='Path to text file that has training data information')
    parser.add_argument('--save-path', type=str, default='./model/saved_model/', help='Where to save the trained model')
    parser.add_argument('--pretrained-model', type=str, help='pretrained model path')
    parser.add_argument('--log-path', type=str, default='/home/ubuntu/dev/LaneNet-Pytorch/logs', help='Where to save scalar data from TensorboardX')
    parser.add_argument(
        '--resume-path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        default=-1,
        nargs='+',
        help='GPUs to use [-1 CPU only] (default: -1)')
    return parser.parse_args()
