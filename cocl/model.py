import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class COCLModel(nn.Module):
    """Classification model f = g ∘ h.

    h: ImageNet-pretrained VGG-16 convolutional layers followed by global
       average pooling (512-dim representation z).
    g: linear classifier.

    No architectural modification is introduced by COCL; the representation
    z is shared by the cross-entropy and contrastive objectives.
    """

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = nn.Sequential(
            vgg16(weights=weights).features,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor):
        z = self.backbone(x)
        logits = self.classifier(z)
        return z, logits
