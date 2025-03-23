import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class FishClassifier(nn.Module):
    def __init__(self):
        super(FishClassifier, self).__init__()
        # Sử dụng weights từ ImageNet
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze các layers ban đầu (tùy chọn)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Thay thế Fully Connected Layer với Dropout
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 11)
        )
        
    def forward(self, x):
        return self.resnet(x)
