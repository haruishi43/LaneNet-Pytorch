import cv2
from torch.utils.data.dataset import Dataset
from torchvision import datasets, models, transforms


class LaneNetDataset(Dataset):
    def __init__(self, text_file, cfg, transform=None):
        # Set image size
        self.height, self.width = cfg.TRAIN.IMG_HEIGHT, cfg.TRAIN.IMG_WIDTH
        
        # Create a list with all image path
        # [[img, binary, instance], ...]
        self.data_locations = []
        with open(text_file, 'r') as f:
            for line in f:
                d = [a for a in line.rstrip('\n').split(' ')]
                self.data_locations.append(d)
                
        self.transform = transform

    def __getitem__(self, index):
        '''Return 3 images (src, binary, instance)'''
        source_path, binary_path, instance_path = self.data_locations[index]
        source_img = cv2.imread(source_path)[:,:,::-1]  # convert to RGB as well
        binary_img = cv2.imread(binary_path, 0)
        instance_img = cv2.imread(instance_path, 0)
        
        # resize images
        source_img = self._resize(source_img, interp=cv2.INTER_LINEAR)
        binary_img = self._resize(binary_img)
        instance_img = self._resize(instance_img)
        
        if self.transform:
            source_img = self.transform(source_img)
        
        return (source_img, binary_img, instance_img)

    def __len__(self):
        return len(self.data_locations)
    
    def _resize(self, img, interp=cv2.INTER_NEAREST):
        '''
        Resize image based on config width and height
        Binary and instance images should not be interpolated with 
        linear algorithms since pixel values are valuable
        '''
        return cv2.resize(img, dsize=(self.width, self.height), interpolation=interp)


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0,'..')
    from config import global_config
    
    # Configs:
    training_file = '/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt'
    CFG = global_config.cfg
    
    ### Print useful stats:
    
    dataset = LaneNetDataset(training_file, CFG)
    print('Number of data: ', len(dataset))
    
    data = dataset[0]  # first data
    print('src image shape: ', data[0].shape)
    print('binary image shape: ', data[1].shape)
    print('instance image shape: ', data[2].shape)
