import torch
from torchvision import transforms

def preprocess_rgb(rgb, cuda=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    
    rgb_tensor = preprocess(rgb)
    
    # have to check if batch or not
    if len(rgb_tensor) != 4:
        rgb_tensor = rgb_tensor.unsqueeze(0)
    
    if cuda:
        rgb_tensor = rgb_tensor.cuda()
    
    return rgb_tensor