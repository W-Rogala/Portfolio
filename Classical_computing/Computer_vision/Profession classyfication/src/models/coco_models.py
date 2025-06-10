"""
COCO pre-trained models implementation for the profession classifier project.
Uses models pretrained on COCO dataset and applies transfer learning.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.models.detection as detection_models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class COCOModelClassifier(nn.Module):
    """
    Classifier based on COCO pre-trained models with custom classification head.
    Uses the backbone of detection models without the detection head.
    """
    def __init__(self, backbone_type='resnet50', num_classes=10, pretrained=True):
        super().__init__()
        self.backbone_type = backbone_type
        
        # Initialize backbone with ImageNet weights (default for torchvision models)
        if backbone_type == 'resnet50':
            # Get ResNet50 backbone from torchvision
            self.backbone = models.resnet50(pretrained=pretrained)
            # Replace last fully connected layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        
        elif backbone_type == 'resnet101':
            # Get ResNet101 backbone from torchvision
            self.backbone = models.resnet101(pretrained=pretrained)
            # Replace last fully connected layer
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
            
        elif backbone_type == 'efficientnet':
            # Get EfficientNet-B0 backbone from torchvision
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            # Replace classifier head
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

    def forward(self, x):
        # Forward pass through the backbone
        return self.backbone(x)


class FasterRCNNFeatureExtractor(nn.Module):
    """
    Feature extractor based on Faster R-CNN pre-trained on COCO.
    Extracts features from the backbone and adds a custom classification head.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()
        
        # Load Faster R-CNN model pre-trained on COCO
        self.model = detection_models.fasterrcnn_resnet50_fpn(
            pretrained=pretrained,
            pretrained_backbone=pretrained
        )
        
        # Freeze backbone layers
        for param in self.model.backbone.parameters():
            param.requires_grad = False
            
        # Use the FPN (Feature Pyramid Network) as feature extractor
        self.feature_extractor = self.model.backbone
        
        # Create a new classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4, 512),  # FPN has 4 feature maps with 256 channels each
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Extract features from FPN
        features = self.feature_extractor(x)
        
        # Process each feature map from FPN
        p2, p3, p4, p5 = features['0'], features['1'], features['2'], features['3']
        
        # Apply average pooling to each feature map
        p2 = self.avgpool(p2)
        p3 = self.avgpool(p3)
        p4 = self.avgpool(p4)
        p5 = self.avgpool(p5)
        
        # Concatenate pooled features
        concat_features = torch.cat([p2, p3, p4, p5], dim=1)
        
        # Pass through classifier
        return self.classifier(concat_features)


def coco_classifier(backbone_type='resnet50', num_classes=10, pretrained=True):
    """
    Create a classifier based on COCO pre-trained backbone.
    
    Args:
        backbone_type (str): Type of backbone to use ('resnet50', 'resnet101', 'efficientnet')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: COCO-based classifier model
    """
    return COCOModelClassifier(backbone_type, num_classes, pretrained)


def faster_rcnn_classifier(num_classes=10, pretrained=True):
    """
    Create a classifier based on Faster R-CNN features.
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        nn.Module: Faster R-CNN based classifier model
    """
    return FasterRCNNFeatureExtractor(num_classes, pretrained)


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test COCO-based models
    models = {
        "COCO-ResNet50": coco_classifier('resnet50', pretrained=False),
        "COCO-ResNet101": coco_classifier('resnet101', pretrained=False),
        "COCO-EfficientNet": coco_classifier('efficientnet', pretrained=False),
        "Faster-RCNN-Features": faster_rcnn_classifier(pretrained=False)
    }
    
    # Print model architecture and parameter counts
    for name, model in models.items():
        try:
            X = torch.randn(size=(1, 3, 224, 224))
            output = model(X)
            
            print(f"\n{name} Architecture:")
            print(f"Input shape: {X.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Trainable parameter count: {count_parameters(model):,}")
        except Exception as e:
            print(f"\n{name} Error: {e}")
