import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)

class MultiTaskFashionModel(nn.Module):

    def __init__(self, num_gender_classes, num_article_classes, num_color_classes, num_usage_classes, freeze_backbone=True):

        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        feature_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.gender_head = self._create_head(feature_size, num_gender_classes)
        self.article_head = self._create_head(feature_size, num_article_classes)
        self.color_head = self._create_head(feature_size, num_color_classes)
        self.usage_head = self._create_head(feature_size, num_usage_classes)

    def _create_head(self, input_features, output_classes):
        return nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_classes)
        )

    def forward(self, x):
        features = self.backbone(x)

        return {
            "gender": self.gender_head(features),
            "article": self.article_head(features),
            "color": self.color_head(features),
            "usage": self.usage_head(features)
        }