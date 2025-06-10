"""
VGG neural network models for the profession classifier project.
"""

import torch
import torch.nn as nn


def vgg_block(num_convs, in_channels, out_channels):
    """
    Create a VGG block with a specified number of convolutional layers.
    
    Args:
        num_convs (int): Number of convolutional layers
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        
    Returns:
        nn.Sequential: A VGG block
    """
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def create_vgg(conv_arch, num_classes=10):
    """
    Create a VGG model with a specified architecture.
    
    Args:
        conv_arch (list): List of tuples (num_convs, out_channels)
        num_classes (int): Number of output classes
        
    Returns:
        nn.Sequential: A VGG model
    """
    conv_blks = []
    in_channels = 3  # RGB images
    
    # Convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # Fully-connected part
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, num_classes)
    )


def vgg_11(num_classes=10):
    """
    Create a VGG-11 model.
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        nn.Sequential: A VGG-11 model
    """
    # VGG-11 architecture
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    return create_vgg(conv_arch, num_classes)


def vgg_11_small(ratio=4, num_classes=10):
    """
    Create a smaller VGG-11 model by reducing the number of channels.
    
    Args:
        ratio (int): Ratio to divide the number of channels by
        num_classes (int): Number of output classes
        
    Returns:
        nn.Sequential: A smaller VGG-11 model
    """
    # VGG-11 architecture
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    # Reduce the number of channels
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    return create_vgg(small_conv_arch, num_classes)


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
    # Test VGG models
    vgg = vgg_11()
    vgg_small = vgg_11_small()
    
    # Print model architecture and parameter count
    print("VGG-11 Architecture:")
    X = torch.randn(size=(1, 3, 224, 224))
    for layer_idx, layer in enumerate(vgg):
        X = layer(X)
        print(f"Layer {layer_idx} ({layer.__class__.__name__}): Output shape {X.shape}")
    
    print(f"\nVGG-11 parameter count: {count_parameters(vgg):,}")
    print(f"Small VGG-11 parameter count: {count_parameters(vgg_small):,}")