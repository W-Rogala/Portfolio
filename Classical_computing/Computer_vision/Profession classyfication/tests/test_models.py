"""
Test model functionality.
"""

import pytest
import torch

from src.models.vgg import vgg_11, vgg_11_small
from src.models.resnet import resnet18


def test_vgg_11():
    """Test VGG-11 model."""
    # Test with default parameters
    model = vgg_11()
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    
    # Check output shape
    assert y.shape == (1, 10)


def test_vgg_11_small():
    """Test small VGG-11 model."""
    # Test with default parameters
    model = vgg_11_small()
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    
    # Check output shape
    assert y.shape == (1, 10)


def test_resnet18():
    """Test ResNet-18 model."""
    # Test with default parameters
    model = resnet18()
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    
    # Check output shape
    assert y.shape == (1, 10)


def test_model_parameter_count():
    """Test model parameter counts."""
    # Create models
    vgg = vgg_11()
    vgg_small = vgg_11_small()
    resnet = resnet18()
    
    # Count parameters
    vgg_params = sum(p.numel() for p in vgg.parameters() if p.requires_grad)
    vgg_small_params = sum(p.numel() for p in vgg_small.parameters() if p.requires_grad)
    resnet_params = sum(p.numel() for p in resnet.parameters() if p.requires_grad)
    
    # Check parameter counts
    assert vgg_params > vgg_small_params
    print(f"VGG-11 parameters: {vgg_params:,}")
    print(f"Small VGG-11 parameters: {vgg_small_params:,}")
    print(f"ResNet-18 parameters: {resnet_params:,}")


if __name__ == "__main__":
    test_vgg_11()
    print("test_vgg_11: PASSED")
    
    test_vgg_11_small()
    print("test_vgg_11_small: PASSED")
    
    test_resnet18()
    print("test_resnet18: PASSED")
    
    test_model_parameter_count()
    print("test_model_parameter_count: PASSED")