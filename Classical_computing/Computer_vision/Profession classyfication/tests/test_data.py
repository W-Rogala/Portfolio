"""
Test data processing functionality.
"""

import os
import pytest
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src.data.preprocess import get_data_loaders, get_augmented_data_loaders, get_class_names


def test_get_class_names():
    """Test get_class_names function."""
    # Create mock dataset structure
    os.makedirs('tests/data/train/class1', exist_ok=True)
    os.makedirs('tests/data/train/class2', exist_ok=True)
    os.makedirs('tests/data/test/class1', exist_ok=True)
    os.makedirs('tests/data/test/class2', exist_ok=True)
    
    # Test function
    class_names = get_class_names('tests/data')
    
    # Check results
    assert len(class_names) == 2
    assert class_names[0] == 'class1'
    assert class_names[1] == 'class2'


def test_get_data_loaders():
    """Test get_data_loaders function."""
    # Skip if no dataset available
    if not os.path.exists('data/idenprof'):
        pytest.skip("Dataset not available")
    
    # Test function
    train_loader, test_loader = get_data_loaders('data/idenprof', batch_size=4)
    
    # Check results
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    
    # Check batch
    images, labels = next(iter(train_loader))
    assert images.shape[0] == 4
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224
    assert labels.shape[0] == 4


def test_get_augmented_data_loaders():
    """Test get_augmented_data_loaders function."""
    # Skip if no dataset available
    if not os.path.exists('data/idenprof'):
        pytest.skip("Dataset not available")
    
    # Test function
    train_loader, test_loader = get_augmented_data_loaders(
        'data/idenprof',
        batch_size=4,
        rotation=30,
        hue=0.05,
        saturation=0.05
    )
    
    # Check results
    assert isinstance(train_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    
    # Check batch
    images, labels = next(iter(train_loader))
    assert images.shape[0] == 4
    assert images.shape[1] == 3
    assert images.shape[2] == 224
    assert images.shape[3] == 224
    assert labels.shape[0] == 4


if __name__ == "__main__":
    test_get_class_names()
    print("test_get_class_names: PASSED")
    
    try:
        test_get_data_loaders()
        print("test_get_data_loaders: PASSED")
    except pytest.skip.Exception as e:
        print(f"test_get_data_loaders: SKIPPED - {e}")
    
    try:
        test_get_augmented_data_loaders()
        print("test_get_augmented_data_loaders: PASSED")
    except pytest.skip.Exception as e:
        print(f"test_get_augmented_data_loaders: SKIPPED - {e}")