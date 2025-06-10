"""
Data preprocessing module for the profession classifier project.
"""

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size=64, image_size=224, shuffle=True):
    """
    Create data loaders for training and testing without augmentation.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for the data loaders
        image_size (int): Size to resize the images to
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        tuple: (train_loader, test_loader) - PyTorch data loaders for training and testing
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader


def get_augmented_data_loaders(data_dir, batch_size=64, image_size=224, 
                              rotation=30, hue=0.05, saturation=0.05, 
                              shuffle=True):
    """
    Create data loaders for training and testing with data augmentation.
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for the data loaders
        image_size (int): Size to resize the images to
        rotation (int): Maximum rotation angle for random rotation
        hue (float): Maximum hue jitter factor
        saturation (float): Maximum saturation jitter factor
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        tuple: (train_loader, test_loader) - PyTorch data loaders for training and testing
    """
    # Transform for training data with augmentation
    transform_train = transforms.Compose([
        transforms.ColorJitter(hue=hue, saturation=saturation),
        transforms.RandomRotation(rotation, expand=True),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Transform for testing data without augmentation
    transform_test = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)
    test_set = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform_test)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
    
    return train_loader, test_loader


def get_class_names(data_dir):
    """
    Get the class names from the dataset.
    
    Args:
        data_dir (str): Directory containing the dataset
        
    Returns:
        dict: Dictionary mapping class indices to class names
    """
    train_dir = os.path.join(data_dir, 'train')
    classes = sorted(os.listdir(train_dir))
    return {i: classes[i] for i in range(len(classes))}


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt
    
    parser = argparse.ArgumentParser(description="Test data preprocessing")
    parser.add_argument("--data_dir", type=str, default="data/idenprof",
                      help="Directory containing the dataset")
    args = parser.parse_args()
    
    # Test data loaders
    print("Testing data loaders...")
    train_loader, test_loader = get_data_loaders(args.data_dir, batch_size=4)
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    print(f"Class names: {class_names}")
    
    # Display a batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # Create a grid of images
    plt.figure(figsize=(10, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        # Un-normalize
        img = images[i] / 2 + 0.5
        plt.imshow(img.permute(1, 2, 0))
        plt.title(class_names[labels[i].item()])
        plt.axis('off')
    
    plt.savefig('sample_batch.png')
    print("Sample batch saved to 'sample_batch.png'")