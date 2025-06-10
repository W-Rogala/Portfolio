"""ResNet neural network model for the profession classifier project."""

import torch
import torch.nn as nn

# ResNet implementation will go here

"""
ResNet neural network model for the profession classifier project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    """
    Residual block for ResNet.
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, 
                              padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # 1x1 convolution for dimension matching if needed
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, 
                                  kernel_size=1, stride=strides)
        else:
            self.conv3 = None
            
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        if self.conv3:
            X = self.conv3(X)
            
        Y += X
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    Create a ResNet block with a specified number of residual units.
    
    Args:
        input_channels (int): Number of input channels
        num_channels (int): Number of output channels
        num_residuals (int): Number of residual units
        first_block (bool): Whether this is the first block
        
    Returns:
        list: A list of residual units
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, 
                               use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


def resnet18(num_classes=10):
    """
    Create a ResNet-18 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        nn.Sequential: A ResNet-18 model
    """
    # First layer
    b1 = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # Residual blocks
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
    b3 = nn.Sequential(*resnet_block(64, 128, 2))
    b4 = nn.Sequential(*resnet_block(128, 256, 2))
    b5 = nn.Sequential(*resnet_block(256, 512, 2))
    
    # Final layers
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
    
    return net


def count_parameters(model):
    """
    Count the number of parameters in a model.
    
    Args:
        model (torch.nn.Module): A PyTorch model
        
    Returns:
        int: Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test ResNet model
    resnet = resnet18()
    
    # Print model architecture and parameter count
    print("ResNet-18 Architecture:")
    X = torch.randn(size=(1, 3, 224, 224))
    for layer_idx, layer in enumerate(resnet):
        X = layer(X)
        print(f"Layer {layer_idx} ({layer.__class__.__name__}): Output shape {X.shape}")
    
    print(f"\nResNet-18 parameter count: {count_parameters(resnet):,}")