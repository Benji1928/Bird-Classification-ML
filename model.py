import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as td
import torchvision as tv
from torchvision import models
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from utils import show_img, Classifier

class VGG16(Classifier):
    def __init__(self, num_classes, fine_tuning=False, dropout_p=0.5):
        super().__init__()
        # Load pretrained VGG16 with batchnorm
        vgg = tv.models.vgg16_bn(pretrained=True)
        
        # Freeze parameters if not fine-tuning
        # for param in vgg.parameters():
           # param.requires_grad = fine_tuning


        for name, param in vgg.features.named_parameters():
            layer_idx = int(name.split('.')[0])  # extract the index before the dot
            if layer_idx in [0, 3, 7]:          # only freeze these specific layers
                param.requires_grad = False


        
        self.features = vgg.features

        # Droupout Value
        dropout_p = 0.5
        
        # Customize classifier
        num_ftrs = vgg.classifier[0].in_features if hasattr(vgg.classifier[0], 'in_features') else 25088
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        f = self.features(x).view(x.shape[0], -1)
        y = self.classifier(f)
        return y


class Resnet18Transfer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = True  # fine-tune all layers

        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),  # reduce overfitting
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)