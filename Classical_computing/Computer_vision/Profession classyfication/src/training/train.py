"""
Enhanced model training module with early stopping and Optuna integration.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.data.download import download_and_extract_idenprof
from src.data.preprocess import get_data_loaders, get_augmented_data_loaders, get_class_names
from src.models.vgg import vgg_11, vgg_11_small
from src.models.resnet import resnet18
from src.models.unet import unet
from src.models.vit import vit_tiny, vit_small, vit_base
from src.models.coco_models import coco_classifier, faster_rcnn_classifier
from src.models.kan import add_kan_layer
from src.utils.metrics import Accumulator, accuracy, evaluate_accuracy


def init_weights(m):
    """
    Initialize weights for the model.
    
    Args:
        m (torch.nn.Module): Module to initialize
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class EarlyStopping:
    """
    Early stopping handler to stop training when validation performance stops improving.
    
    Args:
        patience (int): Number of epochs to wait after the best validation performance
        delta (float): Minimum change in the monitored quantity to qualify as improvement
        mode (str): One of {'min', 'max'}. If 'min', the quantity is to be minimized; if 'max', it is to be maximized
    """
    def __init__(self, patience=5, delta=0, mode='max'):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        # Mode settings
        if mode == 'min':
            self.monitor_op = lambda a, b: a < b - delta
        elif mode == 'max':
            self.monitor_op = lambda a, b: a > b + delta
        else:
            raise ValueError(f"Mode should be one of 'min', 'max'. Got {mode}")
    
    def __call__(self, score):
        """
        Update early stopping state.
        
        Args:
            score (float): Current score to evaluate
            
        Returns:
            bool: True if the monitored quantity improved
        """
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.monitor_op(score, self.best_score):
            # Score improved
            self.best_score = score
            self.counter = 0
            return True
        else:
            # Score didn't improve
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def train_epoch(net, train_iter, loss, optimizer, device, epoch, epochs):
    """
    Train the model for one epoch.
    
    Args:
        net (torch.nn.Module): The network to train
        train_iter (DataLoader): Training data loader
        loss (function): Loss function
        optimizer (Optimizer): Optimizer
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        epochs (int): Total number of epochs
        
    Returns:
        tuple: (train_loss, train_acc) - Training loss and accuracy
    """
    # Set train mode
    net.train()
    
    # Metrics: sum of training loss, sum of training accuracy, number of examples
    metric = Accumulator(3)
    
    # Training loop
    for i, (X, y) in enumerate(train_iter):
        # Move data to device
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        y_hat = net(X)
        l = loss(y_hat, y)
        
        # Backward pass
        l.backward()
        optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            
        # Print progress
        if (i + 1) % (len(train_iter) // 5) == 0 or i == len(train_iter) - 1:
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_iter)}], '
                 f'Loss: {train_l:.4f}, Accuracy: {train_acc:.4f}')
    
    return metric[0] / metric[2], metric[1] / metric[2]


def train(net, train_iter, test_iter, loss, optimizer, num_epochs, device, 
         save_path=None, early_stopping=False, patience=5, trial=None):
    """
    Train the model with optional early stopping and Optuna trial pruning.
    
    Args:
        net (torch.nn.Module): The network to train
        train_iter (DataLoader): Training data loader
        test_iter (DataLoader): Testing data loader
        loss (function): Loss function
        optimizer (Optimizer): Optimizer
        num_epochs (int): Number of epochs to train
        device (torch.device): Device to train on
        save_path (str): Path to save the model
        early_stopping (bool): Whether to use early stopping
        patience (int): Patience for early stopping
        trial (optuna.trial.Trial): Optuna trial for pruning
        
    Returns:
        tuple: (history, best_acc) - Training history and best test accuracy
    """
    # Initialize weights
    net.apply(init_weights)
    
    # Move model to device
    net.to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    # Best test accuracy
    best_acc = 0.0
    
    # Early stopping handler
    if early_stopping:
        early_stopping_handler = EarlyStopping(patience=patience, mode='max')
    
    # Start timer
    start_time = time.time()
    
    # Training loop
    for epoch in range(num_epochs):
        # Train the model for one epoch
        train_loss, train_acc = train_epoch(net, train_iter, loss, optimizer, device, epoch, num_epochs)
        
        # Evaluate the model
        test_acc = evaluate_accuracy(net, test_iter, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Print epoch results
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, '
             f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
        # Optuna pruning
        if trial is not None:
            trial.report(test_acc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            # Create directory if it doesn't exist
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save model
                torch.save(net.state_dict(), save_path)
                print(f'Model saved to {save_path}')
        
        # Early stopping
        if early_stopping:
            improvement = early_stopping_handler(test_acc)
            if early_stopping_handler.early_stop:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Print training time
    total_time = time.time() - start_time
    print(f'Training time: {total_time:.2f}s, {total_time/num_epochs:.2f}s/epoch')
    
    return history, best_acc


def plot_history(history, title, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Training history
        title (str): Plot title
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Set title
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save plot
    if save_path:
        plt.savefig(save_path)
        print(f'Plot saved to {save_path}')
    
    plt.show()


def get_model(model_name, num_classes=10, use_kan=False, kan_hidden_sizes=None, kan_width=16):
    """
    Get a model by name, optionally with KAN layers.
    
    Args:
        model_name (str): Model name
        num_classes (int): Number of output classes
        use_kan (bool): Whether to use KAN layers
        kan_hidden_sizes (list): List of hidden layer sizes for KAN network
        kan_width (int): Width of KAN layers
        
    Returns:
        torch.nn.Module: Model
    """
    # Default KAN hidden sizes
    if kan_hidden_sizes is None:
        kan_hidden_sizes = [128, 64]
    
    # Create base model
    if model_name == 'vgg':
        model = vgg_11(num_classes=num_classes)
    elif model_name == 'vgg_small':
        model = vgg_11_small(num_classes=num_classes)
    elif model_name == 'resnet':
        model = resnet18(num_classes=num_classes)
    elif model_name == 'unet':
        model = unet(n_classes=num_classes)
    elif model_name == 'vit_tiny':
        model = vit_tiny(num_classes=num_classes)
    elif model_name == 'vit_small':
        model = vit_small(num_classes=num_classes)
    elif model_name == 'vit_base':
        model = vit_base(num_classes=num_classes)
    elif model_name == 'coco_resnet50':
        model = coco_classifier(backbone_type='resnet50', num_classes=num_classes)
    elif model_name == 'coco_resnet101':
        model = coco_classifier(backbone_type='resnet101', num_classes=num_classes)
    elif model_name == 'coco_efficientnet':
        model = coco_classifier(backbone_type='efficientnet', num_classes=num_classes)
    elif model_name == 'faster_rcnn':
        model = faster_rcnn_classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Apply KAN if selected
    if use_kan:
        model = add_kan_layer(
            model, 
            kan_hidden_sizes=kan_hidden_sizes, 
            num_classes=num_classes,
            kan_width=kan_width
        )
    
    return model


def main(args):
    """
    Main function.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f'Using device: {device}')
    
    # Download and extract dataset if needed
    data_dir = download_and_extract_idenprof(args.data_dir)
    
    # Get class names
    class_names = get_class_names(data_dir)
    num_classes = len(class_names)
    print(f'Number of classes: {num_classes}')
    
    # Get data loaders
    if args.augment:
        print('Using data augmentation')
        train_iter, test_iter = get_augmented_data_loaders(
            data_dir, 
            batch_size=args.batch_size, 
            image_size=args.image_size,
            rotation=args.rotation,
            hue=args.hue,
            saturation=args.saturation
        )
    else:
        print('Not using data augmentation')
        train_iter, test_iter = get_data_loaders(
            data_dir, 
            batch_size=args.batch_size, 
            image_size=args.image_size
        )
    
    # Get model
    model = get_model(
        args.model, 
        num_classes, 
        use_kan=args.use_kan,
        kan_hidden_sizes=[args.kan_hidden1, args.kan_hidden2],
        kan_width=args.kan_width
    )
    print(f'Using model: {args.model}')
    
    # Count model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {trainable_params:,} trainable parameters')
    
    # Loss function
    loss = nn.CrossEntropyLoss()
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Model save path
    model_name = f"{args.model}_{'augmented' if args.augment else 'standard'}"
    if args.use_kan:
        model_name += "_kan"
    save_path = os.path.join(args.save_dir, f"{model_name}.pth")
    
    # Train model
    history, best_acc = train(
        model, 
        train_iter, 
        test_iter, 
        loss, 
        optimizer, 
        args.epochs, 
        device, 
        save_path,
        early_stopping=args.early_stopping,
        patience=args.patience
    )
    
    # Plot history
    plot_path = os.path.join(args.save_dir, f"{model_name}_history.png")
    title = f"{args.model.upper()} {'with' if args.augment else 'without'} Augmentation"
    if args.use_kan:
        title += " with KAN"
    plot_history(history, title, plot_path)
    
    print(f'Best test accuracy: {best_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for profession classification")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to save the dataset')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Size to resize the images to')
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    
    # Augmentation parameters
    parser.add_argument('--augment', action='store_true',
                      help='Use data augmentation')
    parser.add_argument('--rotation', type=int, default=30,
                      help='Maximum rotation angle for augmentation')
    parser.add_argument('--hue', type=float, default=0.05,
                      help='Maximum hue jitter factor for augmentation')
    parser.add_argument('--saturation', type=float, default=0.05,
                      help='Maximum saturation jitter factor for augmentation')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='vgg',
                      choices=['vgg', 'vgg_small', 'resnet', 'unet', 
                              'vit_tiny', 'vit_small', 'vit_base',
                              'coco_resnet50', 'coco_resnet101', 'coco_efficientnet', 'faster_rcnn'],
                      help='Model architecture')
    
    # KAN parameters
    parser.add_argument('--use_kan', action='store_true',
                      help='Use KAN layers')
    parser.add_argument('--kan_hidden1', type=int, default=128,
                      help='Size of first hidden layer in KAN')
    parser.add_argument('--kan_hidden2', type=int, default=64,
                      help='Size of second hidden layer in KAN')
    parser.add_argument('--kan_width', type=int, default=16,
                      help='Width of KAN layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=40,
                      help='Number of epochs to train')
    parser.add_argument('--optimizer', type=str, default='sgd',
                      choices=['sgd', 'adam', 'adamw'],
                      help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.05,
                      help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum (for SGD)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help='Weight decay')
    parser.add_argument('--early_stopping', action='store_true',
                      help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                      help='Patience for early stopping')
    parser.add_argument('--no_cuda', action='store_true',
                      help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Save parameters
    parser.add_argument('--save_dir', type=str, default='models',
                      help='Directory to save models and plots')
    
    args = parser.parse_args()
    main(args)
