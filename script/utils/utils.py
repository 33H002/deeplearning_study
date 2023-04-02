import torch
import torchvision

from PIL import Image

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, dataset, phase='train', transform=None):
        self.dataset = dataset
        self.phase = phase  
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        image = Image.open(self.dataset[item][0]).convert('RGB')
        label = self.dataset[item][1]
        if self.transform:
            image = self.transform(image, self.phase)
        return image, label
    
    
class TransformsCE:
    def __init__(self, size):
        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.data_transform = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=size), 
                torchvision.transforms.RandAugment(),     
                torchvision.transforms.ToTensor(),
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]),
            'test': torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]),
        }

    def __call__(self, x, phase='train'):
        return self.data_transform[phase](x)