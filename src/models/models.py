import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class MultiTaskFashionModel(nn.Module):

    def __init__(
        self,
        num_gender_classes,
        num_article_classes,
        num_color_classes,
        num_usage_classes,
        freeze_backbone=True,
        pretrained_backbone=True,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained_backbone else None
        base_model = resnet18(weights=weights)

        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        self.initial_layers = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,  
        )

        self.remaining_layers = nn.Sequential(
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool,
        )

        self.color_pool = nn.AdaptiveAvgPool2d((1, 1))

        final_feature_size = 512

        self.gender_head = self._create_head(final_feature_size, num_gender_classes)
        self.article_head = self._create_head(final_feature_size, num_article_classes)
        self.usage_head = self._create_head(final_feature_size, num_usage_classes)

        self.color_head = self._create_head(64, num_color_classes)

    def _create_head(self, input_features, output_classes):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_classes),
        )

    def forward(self, x):
        early_features = self.initial_layers(x)

        final_features = self.remaining_layers(early_features)
        final_features = torch.flatten(final_features, 1)

        color_features = self.color_pool(early_features)
        color_features = torch.flatten(color_features, 1)

        return {
            "gender": self.gender_head(final_features),
            "article": self.article_head(final_features),
            "usage": self.usage_head(final_features),
            "color": self.color_head(color_features),
        }