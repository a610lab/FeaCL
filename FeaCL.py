import os

os.environ['CUDA_VISIBLE_DEVICES'] ='1'

import torch
import torch.nn as nn
import copy
import torchvision.transforms as transforms
import torch.nn.functional as F
import torchvision
import numpy as np
import pytorch_lightning as pl
import lightly
import lightly.data
from lightly.data.dataset import LightlyDataset
from lightly.data.lightly_subset import LightlySubset
from lightly.models.modules import BYOLPredictionHead, BarlowTwinsProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from loss import CrossCorrelationMatrix, BarlowTwinsLossMY
from utils import BenchmarkModule
from lightly.loss import NegativeCosineSimilarity, BarlowTwinsLoss


import timm
import random
from functools import partial
from MyAugmentation_Triple import MyCollateFunction

num_workers = 0
max_epochs = 300
knn_k = 200
knn_t = 0.1
classes = 3
batch_size = 16  
seed = 1
samples_rate = 1.0  # TODO 训练用数据集的比例


pl.seed_everything(seed)

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0
device = 'cuda:0' if gpus else 'cpu'

collate_fn = MyCollateFunction()  # TODO  choose weak/strong augmentation

val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([144, 96]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize([144, 96]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# 数据集加载 todo
path_to_train = "/home/a610/Project/licheng/Zhongnan_plaque/train-val"
path_to_test = "/home/a610/Project/licheng/Zhongnan_plaque/test"

# 获取数据集的子集
random.seed(seed)
base_dataset = LightlyDataset(input_dir=path_to_train, transform=val_transforms)
filenames_base_dataset = base_dataset.get_filenames()

no_samples_subset = int(len(filenames_base_dataset) * samples_rate)  # 获取子集占数据集的比例
filenames_subset = random.sample(filenames_base_dataset, no_samples_subset)

subset = LightlySubset(base_dataset=base_dataset, filenames_subset=filenames_subset)

# 用于分类器训练的数据集
dataset_train_classifier = subset

# 用于自监督训练的数据集
dataset_train_ssl = lightly.data.LightlyDataset(
    input_dir=path_to_train, transform=torchvision.transforms.Resize([144, 96]))
# 用于KNN训练的数据集
dataset_train_kNN = lightly.data.LightlyDataset(
    input_dir=path_to_train, transform=test_transforms)
# 测试数据集
dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_test, transform=test_transforms)

dataloader_train_ssl = torch.utils.data.DataLoader(
    dataset_train_ssl,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)
dataloader_train_kNN = torch.utils.data.DataLoader(
    dataset_train_kNN,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers
)
dataloader_train_classifier = torch.utils.data.DataLoader(
    dataset_train_classifier,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)
dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

def fn(warmup_steps, step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    else:
        return 1.0

def linear_warmup_decay(warmup_steps):
    return partial(fn, warmup_steps)


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class FeaCL(BenchmarkModule):
    def __init__(self, dataloader_kNN, gpus, classes, knn_k, knn_t):
        super().__init__(dataloader_kNN, gpus, classes, knn_k, knn_t)

        # create a ResNet backbone and remove the classification head
        resnest = timm.create_model('resnest50d')  # TODO
        self.warmup_epochs = 10
        self.num_training_samples = len(dataset_train_ssl)
        self.train_iters_per_epoch = self.num_training_samples // batch_size

        self.backbone = nn.Sequential(
            *list(resnest.children())[:-1],
        )

        self.projection_head_s = BarlowTwinsProjectionHead(2048, 2048, 2048)  
        self.prediction_head_s = BYOLPredictionHead(2048, 4096, 256)        


        self.backbone_momentum_t1 = copy.deepcopy(self.backbone)                    
        self.projection_head_momentum_t1 = copy.deepcopy(self.projection_head_s)    
        self.prediction_head_momentum_t1 = copy.deepcopy(self.prediction_head_s)    

        deactivate_requires_grad(self.backbone_momentum_t1)
        deactivate_requires_grad(self.projection_head_momentum_t1)
        deactivate_requires_grad(self.prediction_head_momentum_t1)

        self.criterion  = BarlowTwinsLoss()
        self.criterion2 = NegativeCosineSimilarity()
        self.criterion3 = BarlowTwinsLossMY(device=device)

        # self.criterion3 = loss_fn()

    # student1
    def forward_s1(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z1 = self.projection_head_s(y)
        s1 = self.prediction_head_s(z1)
        return z1, s1

    # student2
    def forward_s2(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z2 = self.projection_head_s(y)
        s2 = self.prediction_head_s(z2)
        return z2, s2

    # teacher1
    def forward1_momentum(self, x):
        y = self.backbone_momentum_t1(x).flatten(start_dim=1)
        z3 = self.projection_head_momentum_t1(y)
        t1 = self.prediction_head_momentum_t1(z3)
        t1 = t1.detach()
        return z3, t1

    def training_step(self, batch, batch_idx):

        
        momentum1 = 0.996
        update_momentum(self.backbone, self.backbone_momentum_t1, m=momentum1)
        update_momentum(self.projection_head_s, self.projection_head_momentum_t1, m=momentum1)
        update_momentum(self.prediction_head_s, self.prediction_head_momentum_t1, m=momentum1)

        (x, x0, x1), _, _ = batch
        S0 = self.forward_s1(x)[0]          
        P1 = self.forward_s1(x)[1]          
        S1 = self.forward_s2(x0)[0]         
        P2 = self.forward_s2(x0)[1]         
        T1 = self.forward1_momentum(x1)[0]  
        P3 = self.forward1_momentum(x1)[1]  

        ccm_xw = CrossCorrelationMatrix(T1, S0)
        ccm_xs = CrossCorrelationMatrix(T1, S1)
        ccm_ws = CrossCorrelationMatrix(S0, S1)
        ccm_all = torch.mul(ccm_xw, 0.1) + torch.mul(ccm_xs, 0.2) + torch.mul(ccm_ws, 0.7)
        loss1 = self.criterion3(ccm_all, S0)

        loss2 = self.criterion2(P3, P1) + self.criterion2(P3, P2)

        loss = loss1 + loss2
        self.log('train_loss_ssl', loss)
        return loss

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

if __name__ == '__main__':

    model = FeaCL(dataloader_train_kNN, gpus=gpus, classes=classes, knn_k=knn_k, knn_t=knn_t)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator='gpu', devices=1, enable_progress_bar=True)  
    trainer.fit(
        model,
        train_dataloaders=dataloader_train_ssl,
        val_dataloaders=dataloader_test
    )

    state_dict = {
        'resnest50d_parameters': model.backbone.state_dict()
    }
    torch.save(state_dict, './pre_models/TriBT(FeatureMix)_L2_S.pth')
    state_dict1 = {
        'resnest50d_parameters': model.backbone_momentum_t1.state_dict()
    }
    torch.save(state_dict1, './pre_models/TriBT(FeatureMix)_L2_T.pth')

    print(f'Highest test accuracy: {model.max_accuracy:.4f}')
