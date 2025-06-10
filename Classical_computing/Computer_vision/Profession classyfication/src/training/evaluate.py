"""
Evaluation module for the profession classifier project.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from src.data.preprocess import get_data_loaders, get_class_names
from src.models.vgg import vgg_11, vgg_11_small
from src.models.resnet import resnet18
from src.training.train import get_model
from src.utils.metrics import evaluate_accuracy, predict_class


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (dict): Dictionary mapping class indices to class names
        save_path (str): Path to save the plot
    """
    # Get class names list
    labels = [class_names[i] for i in range(len(class_names))]
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save plot
    if save_path:
        plt.savefig(save_path)
        print(f'Confusion matrix saved to {save_path}')
    
    plt.show()


def visualize_predictions(model, data_loader, class_names, device, num_samples=10, save_path=None):
    """
    Visualize model predictions.
    
    Args:
        model (torch.nn.Module): Model
        data_loader (DataLoader): Data loader
        class_names (dict): Dictionary mapping class indices to class names
        device (torch.device): Device
        num_samples (int): Number of samples to visualize
        save_path (str): Path to save the plot
    """
    # Get a batch of images
    dataiter = iter(data_loader)
    images, labels = next(dataiter)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 6))
    for i in range(min(num_samples, len(images))):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        
        # Un-normalize image
        img = images[i] / 2 + 0.5
        plt.imshow(img.permute(1, 2, 0))
        
        # Color-code the prediction
        color = 'green' if preds[i] == labels[i] else 'red'
        ax.set_title(f'True: {class_names[labels[i].item()]}\nPred: {class_names[preds[i]]}', 
                   color=color)
    
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path)
        print(f'Predictions visualization saved to {save_path}')
    
    plt.show()


def evaluate_model(model_path, data_dir, batch_size=64, image_size=224, no_cuda=False):
    """
    Evaluate a model.
    
    Args:
        model_path (str): Path to the saved model
        data_dir (str): Path to the dataset
        batch_size (int): Batch size
        image_size (int): Image size
        no_cuda (bool): Whether to disable CUDA
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Get class names
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)
    print(f'Number of classes: {num_classes}')
    
    # Get data loaders
    _, test_loader = get_data_loaders(data_dir, batch_size, image_size)
    
    # Determine model type from filename
    model_name = os.path.basename(model_path).split('_')[0]
    print(f'Model type: {model_name}')
    
    # Get model
    model = get_model(model_name, num_classes)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # Evaluate model
    test_acc = evaluate_accuracy(model, test_loader, device)
    print(f'Test accuracy: {test_acc:.4f}')
    
    # Get all predictions
    y_true, y_pred = [], []
    for images, labels in test_loader:
        images, labels = images.to(device), labels
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
    
    # Print classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=[class_names[i] for i in range(num_classes)])
    print('\nClassification Report:')
    print(report)
    
    # Plot confusion matrix
    save_dir = os.path.dirname(model_path)
    model_basename = os.path.basename(model_path).split('.')[0]
    cm_path = os.path.join(save_dir, f"{model_basename}_confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, cm_path)
    
    # Visualize predictions
    viz_path = os.path.join(save_dir, f"{model_basename}_predictions.png")
    visualize_predictions(model, test_loader, class_names, device, 10, viz_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the saved model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/idenprof',
                       help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for evaluation')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Size to resize the images to')
    
    # Device parameters
    parser.add_argument('--no_cuda', action='store_true',
                      help='Disable CUDA')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.batch_size, args.image_size, args.no_cuda)