"""
Visualization module for the profession classifier project.
"""

import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from src.data.download import download_and_extract_idenprof
from src.data.preprocess import get_class_names


def visualize_sample(data_dir, num_samples=8, seed=None, save_path=None):
    """
    Visualize random samples from the dataset.
    
    Args:
        data_dir (str): Path to the dataset
        num_samples (int): Number of samples to visualize
        seed (int): Random seed for reproducibility
        save_path (str): Path to save the visualization
    """
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Load dataset
    train_dir = os.path.join(data_dir, 'train')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(train_dir, transform=transform)
    
    # Get class names
    class_names = get_class_names(data_dir)
    
    # Sample indices
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Plot samples
    fig, axes = plt.subplots(2, num_samples//2, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].set_title(class_names[label])
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    if save_path:
        plt.savefig(save_path)
        print(f'Visualization saved to {save_path}')
    
    plt.show()


def visualize_augmentation(data_dir, rotation=30, hue=0.5, saturation=0.5, 
                         image_size=224, seed=None, save_path=None):
    """
    Visualize the effect of data augmentation on random samples.
    
    Args:
        data_dir (str): Path to the dataset
        rotation (int): Maximum rotation angle for augmentation
        hue (float): Maximum hue jitter factor for augmentation
        saturation (float): Maximum saturation jitter factor for augmentation
        image_size (int): Image size
        seed (int): Random seed for reproducibility
        save_path (str): Path to save the visualization
    """
    # Set seed for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Define transforms
    transform_original = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    transform_augmented = transforms.Compose([
        transforms.ColorJitter(hue=hue, saturation=saturation),
        transforms.RandomRotation(rotation, expand=True),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    # Load dataset
    train_dir = os.path.join(data_dir, 'train')
    original_dataset = ImageFolder(train_dir, transform=transform_original)
    
    # Get class names
    class_names = get_class_names(data_dir)
    
    # Sample indices
    indices = random.sample(range(len(original_dataset)), 4)
    
    # Plot samples
    fig, axes = plt.subplots(4, 2, figsize=(10, 15))
    
    for i, idx in enumerate(indices):
        # Get original image
        img_path, label = original_dataset.samples[idx]
        
        # Display original image
        img_original = original_dataset[idx][0]
        axes[i, 0].imshow(img_original.permute(1, 2, 0))
        axes[i, 0].set_title(f"{class_names[label]} (Original)")
        axes[i, 0].axis('off')
        
        # Load image and apply augmentation
        img = Image.open(img_path).convert('RGB')
        img_augmented = transform_augmented(img)
        
        # Display augmented image
        axes[i, 1].imshow(img_augmented.permute(1, 2, 0))
        axes[i, 1].set_title(f"{class_names[label]} (Augmented)")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    if save_path:
        plt.savefig(save_path)
        print(f'Augmentation visualization saved to {save_path}')
    
    plt.show()


def visualize_class_distribution(data_dir, save_path=None):
    """
    Visualize the class distribution in the dataset.
    
    Args:
        data_dir (str): Path to the dataset
        save_path (str): Path to save the visualization
    """
    # Get class names
    class_names = get_class_names(data_dir)
    
    # Count samples per class
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    train_counts = []
    test_counts = []
    
    for class_idx, class_name in class_names.items():
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        train_count = len(os.listdir(train_class_dir))
        test_count = len(os.listdir(test_class_dir))
        
        train_counts.append(train_count)
        test_counts.append(test_count)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax.bar(x - width/2, train_counts, width, label='Train')
    ax.bar(x + width/2, test_counts, width, label='Test')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of samples')
    ax.set_title('Class distribution in the dataset')
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in range(len(class_names))])
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save visualization
    if save_path:
        plt.savefig(save_path)
        print(f'Class distribution visualization saved to {save_path}')
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the dataset")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing the dataset')
    
    # Visualization parameters
    parser.add_argument('--type', type=str, default='sample',
                      choices=['sample', 'augmentation', 'distribution'],
                      help='Type of visualization')
    parser.add_argument('--samples', type=int, default=8,
                      help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Augmentation parameters
    parser.add_argument('--rotation', type=int, default=30,
                      help='Maximum rotation angle for augmentation')
    parser.add_argument('--hue', type=float, default=0.5,
                      help='Maximum hue jitter factor for augmentation')
    parser.add_argument('--saturation', type=float, default=0.5,
                      help='Maximum saturation jitter factor for augmentation')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='visualizations',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Download dataset if needed
    data_dir = download_and_extract_idenprof(args.data_dir)
    
    # Choose visualization type
    if args.type == 'sample':
        save_path = os.path.join(args.save_dir, 'sample_visualization.png')
        visualize_sample(data_dir, args.samples, args.seed, save_path)
    
    elif args.type == 'augmentation':
        save_path = os.path.join(args.save_dir, 'augmentation_visualization.png')
        visualize_augmentation(data_dir, args.rotation, args.hue, args.saturation, 
                             224, args.seed, save_path)
    
    elif args.type == 'distribution':
        save_path = os.path.join(args.save_dir, 'distribution_visualization.png')
        visualize_class_distribution(data_dir, save_path)