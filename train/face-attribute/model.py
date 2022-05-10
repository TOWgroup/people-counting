
import torch.nn as nn
from torchvision import models

class MultiLableModel(nn.Module):
    def __init__(self, num_attr = 40):
        super(MultiLableModel, self).__init__()
        self.base_model = models.resnet50(pretrained=True)

        for params in self.base_model.parameters():
            params.requires_grad = False

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(nn.Dropout(p = 0.2),
                                           nn.Linear(num_ftrs, num_attr, bias=True),
                                           nn.Sigmoid())
        
    def forward(self, x):
        x = self.base_model(x)
        return x   