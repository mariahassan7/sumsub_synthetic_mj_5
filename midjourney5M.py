import timm
import torch
from torch import nn

class Model5M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model('timm/tf_mobilenetv3_large_100.in1k', pretrained=False, num_classes=0)

        self.clf = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2))

    def forward(self, image):
        image_features = self.model(image)
        return self.clf(image_features)