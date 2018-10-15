import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class VGGEncoder(nn.Module):

    def __init__(self, pretrained=True, vgg_type='vgg16_bn'):
        super(VGGEncoder, self).__init__()

        vgg = models.vgg16_bn(pretrained=pretrained)
        features = list(vgg.classifier.children())[:-1]  # remove the last layer
        vgg.classifier = nn.Sequential(*features)
        self.features = vgg.features
        
        # freeze all layers since vgg is used for feature extraction
        for param in self.features.parameters():
            param.require_grad = False
            
        self.eval()

    def forward(self, x):
        layers = ['pool3', 'pool4', 'pool5']
        p = 0
        results = {}
        
        for i, model in enumerate(self.features):
            x = model(x)
            if i in {23,33,43}:  # last 3 pooling layer
                results[layers[p]] = x
                p += 1
        
        return results


if __name__ == '__main__':
    import sys
    sys.path.insert(0,'..')
    from config import global_config
    from dataset import LaneNetDataset
    TRAIN_FILE = '/home/ubuntu/dev/LaneNet-Pytorch/data/training_data/train.txt'
    CFG = global_config.cfg
    
    dataset = LaneNetDataset(TRAIN_FILE, CFG)
    inputs = next(iter(dataset))  # (src, binary, instance)
    
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    
    # preprocess images
    src_tensor = preprocess(inputs[0])
    src_tensor = src_tensor.unsqueeze(0).cuda()
    print('After preprocessing: ', src_tensor.shape)
    
    vgg = VGGEncoder().cuda()
    # print(vgg)
    
    output = vgg(src_tensor)
    
    # outputs of the last 3 pooling layers:
    print(output['pool3'].size())
    print(output['pool4'].size())
    print(output['pool5'].size())
