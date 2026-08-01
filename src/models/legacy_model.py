import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class LegacyResNetFashionModel(nn.Module):

    def __init__(
        self,
        num_gender_classes,
        num_article_classes,
        num_color_classes,
        num_usage_classes,
        freeze_backbone=True,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        final_feature_size = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()  # Remove final linear layer

        # Match the EXACT checkpoint head architecture
        self.gender_head = self._create_head(
            final_feature_size, num_gender_classes
        )
        self.article_head = self._create_head(
            final_feature_size, num_article_classes
        )
        self.usage_head = self._create_head(
            final_feature_size, num_usage_classes
        )
        self.color_head = self._create_head(
            final_feature_size, num_color_classes
        )

    def _create_head(self, input_features, output_classes):
        """Recreates head structure matching saved weights:

        Linear(512, 512) -> ReLU -> Dropout -> Linear(512, output_classes)
        """
        return nn.Sequential(
            nn.Linear(input_features, 512),  # Index 0: weight shape [512, 512]
            nn.ReLU(),  # Index 1
            nn.Dropout(0.2),  # Index 2
            nn.Linear(
                512, output_classes
            ),  # Index 3: weight shape [classes, 512]
        )

    def forward(self, x):
        features = self.backbone(x)

        return {
            "gender": self.gender_head(features),
            "article": self.article_head(features),
            "usage": self.usage_head(features),
            "color": self.color_head(features),
        }