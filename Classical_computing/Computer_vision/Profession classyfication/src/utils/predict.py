"""
Prediction module for the profession classifier project.
"""

import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from src.data.preprocess import get_class_names
from src.models.vgg import vgg_11, vgg_11_small
from src.models.resnet import resnet18
from src.training.train import get_model


def load_model(model_path, model_type, num_classes=10, device='cpu'):
    """
    Load a trained model.
    
    Args:
        model_path (str): Path to the model file
        model_type (str): Type of model ('vgg', 'vgg_small', or 'resnet')
        num_classes (int): Number of output classes
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Get model
    model = get_model(model_type, num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_path, transform=None, device='cpu'):
    """
    Predict the class of an image.
    
    Args:
        model (torch.nn.Module): Model
        image_path (str): Path to the image
        transform (transforms.Compose): Transform to apply to the image
        device (str): Device to use ('cpu' or 'cuda')
        
    Returns:
        tuple: (class_idx, confidence) - Predicted class index and confidence
    """
    # Default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    
    return predicted_class.item(), confidence.item()


def predict_batch(model, image_paths, class_names, transform=None, device='cpu'):
    """
    Predict classes for a batch of images and visualize the results.
    
    Args:
        model (torch.nn.Module): Model
        image_paths (list): List of image paths
        class_names (dict): Dictionary mapping class indices to class names
        transform (transforms.Compose): Transform to apply to the images
        device (str): Device to use ('cpu' or 'cuda')
    """
    # Default transform if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    # Number of images
    n = len(image_paths)
    
    # Create figure
    fig, axes = plt.subplots(1, n, figsize=(n*4, 4))
    if n == 1:
        axes = [axes]
    
    # Predict and visualize
    for i, image_path in enumerate(image_paths):
        # Make prediction
        class_idx, confidence = predict_image(model, image_path, transform, device)
        class_name = class_names[class_idx]
        
        # Load and display image
        image = Image.open(image_path).convert('RGB')
        axes[i].imshow(image)
        axes[i].set_title(f"Prediction: {class_name}\nConfidence: {confidence:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def main(args):
    """
    Main function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Get class names
    class_names = get_class_names(args.data_dir)
    num_classes = len(class_names)
    print(f"Number of classes: {num_classes}")
    
    # Determine model type from filename
    model_name = os.path.basename(args.model_path).split('_')[0]
    print(f"Model type: {model_name}")
    
    # Load model
    model = load_model(args.model_path, model_name, num_classes, device)
    print(f"Model loaded from {args.model_path}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Predict
    if args.image_dir:
        # Get all images in directory
        image_paths = []
        for file in os.listdir(args.image_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(args.image_dir, file))
        
        if not image_paths:
            print(f"No images found in {args.image_dir}")
            return
        
        # Predict and visualize
        predict_batch(model, image_paths, class_names, transform, device)
    
    elif args.image_path:
        # Single image prediction
        class_idx, confidence = predict_image(model, args.image_path, transform, device)
        class_name = class_names[class_idx]
        print(f"Prediction: {class_name} (Confidence: {confidence:.2f})")
        
        # Display image
        image = Image.open(args.image_path).convert('RGB')
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Prediction: {class_name}\nConfidence: {confidence:.2f}")
        plt.axis('off')
        plt.show()
    
    else:
        print("Please provide an image path or directory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict professions from images")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/idenprof',
                       help='Path to the dataset (for class names)')
    
    # Input parameters
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image_path', type=str,
                     help='Path to a single image')
    group.add_argument('--image_dir', type=str,
                     help='Directory containing images to predict')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='cuda',
                      choices=['cpu', 'cuda'],
                      help='Device to use for prediction')
    
    args = parser.parse_args()
    main(args)