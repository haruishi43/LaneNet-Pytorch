import os
import sys
import torch

from arguments import init_args
from config import global_config
from train import train

CFG = global_config.cfg


def main():
    
    # initialize args
    args = init_args()
    
    # device settings:
    if not args.gpu_ids:
        args.main_device = torch.device('cpu')
    else:
        visible_devices = ','.join(map(str, args.gpu_ids))
        print('visible devices: ', visible_devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        
        # gpu ids (manually set)
        args.gpu_ids = [i for i in range(0, len(args.gpu_ids))]
        #print(torch.cuda.device_count())
        
        args.main_device = torch.device(args.gpu_ids[0])
    
    train(args)
    

if __name__ == "__main__":
    
    main()