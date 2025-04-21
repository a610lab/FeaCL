import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from typing import List, Tuple, Union
from randaugment import RandAugmentMC

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

class MyCollateFunction(nn.Module):

    def __init__(self,
                 hf_prob: float = 1.0,
                 vf_prob: float = 1.0,
                 normalize: dict = imagenet_normalize
        ):
        super(MyCollateFunction, self).__init__()

        self.no_augment = transforms.Compose([
            # transforms.Resize([96,144]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
        ])
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(p=hf_prob),
            # transforms.Resize([96,144]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
        ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(p=hf_prob),
            # transforms.Resize([96,144]),
            RandAugmentMC(n=6, m=10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
        ])
        
    
    def forward(self, batch: List[tuple]):
        
        batch_size = len(batch)

        # list of transformed images
        img_weak = [self.weak(batch[i][0]).unsqueeze_(0)
                      for i in range(batch_size)]
        img_strong = [self.strong(batch[i][0]).unsqueeze_(0)
                      for i in range(batch_size)]
        img = [self.no_augment(batch[i][0]).unsqueeze_(0)
                for i in range(batch_size)]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        transforms = (
            torch.cat(img_weak, 0),
            torch.cat(img_strong, 0),
            torch.cat(img, 0),
        )

        return transforms, labels, fnames
