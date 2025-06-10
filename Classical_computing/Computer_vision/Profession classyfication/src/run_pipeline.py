"""
Enhanced training pipeline with advanced models and Optuna optimization.
"""

import os
import time
import argparse
from datetime import datetime

from src.data.download import download_and_extract_idenprof
from src.training.train import main as train_main
from src.training.evaluate import evaluate_model
from src.training.optuna_optimization import run_hyperparameter_optimization, get_best_model_from_study
from src.utils.visualization import (
    visualize_sample, 
    visualize_augmentation,
    visualize_class_distribution
)


class PipelineArgs:
    """Class to simulate argparse namespace for different pipeline stages."""
    pass


def create_visualizations(data_dir, save_dir):
    """
    Create visualizations of the dataset.
    
    Args:
        data_dir (str): Path to the dataset
        save_dir (str): Directory to save visualizations
    """
    print("\n=== Creating Visualizations ===")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Sample visualization
    print("Creating sample visualization...")
    visualize_sample(data_dir, num_samples=8, seed=42, 
                    save_path=os.path.join(save_dir, "sample_visualization.png"))
    
    # Augmentation visualization
    print("Creating augmentation visualization...")
    visualize_augmentation(data_dir, rotation=30, hue=0.5, saturation=0.5, 
                         seed=42, 
                         save_path=os.path.join(save_dir, "augmentation_visualization.png"))
    
    # Class distribution visualization
    print("Creating class distribution visualization...")
    visualize_class_distribution(data_dir, 
                              save_path=os.path.join(save_dir, "class_distribution.png"))


def train_models(data_dir, save_dir, args):
    """
    Train models with and without data augmentation.
    
    Args:
        data_dir (str): Path to the dataset
        save_dir (str): Directory to save models
        args (argparse.Namespace): Command line arguments
    """
    print("\n=== Training Models ===")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get models to train
    models = args.models.split(',')
    
    # Train each model
    for model_name in models:
        print(f"\nTraining {model_name.upper()}...")
        
        # Standard training
        print(f"\nTraining {model_name.upper()} without augmentation...")
        
        # Create args for training
        train_args = PipelineArgs()
        train_args.data_dir = data_dir
        train_args.save_dir = save_dir
        train_args.model = model_name
        train_args.batch_size = args.batch_size
        train_args.image_size = args.image_size
        train_args.lr = args.lr
        train_args.epochs = args.epochs
        train_args.optimizer = args.optimizer
        train_args.momentum = args.momentum
        train_args.weight_decay = args.weight_decay
        train_args.seed = args.seed
        train_args.no_cuda = args.no_cuda
        train_args.augment = False
        train_args.rotation = args.rotation
        train_args.hue = args.hue
        train_args.saturation = args.saturation
        train_args.early_stopping = args.early_stopping
        train_args.patience = args.patience
        train_args.use_kan = args.use_kan
        train_args.kan_hidden1 = args.kan_hidden1
        train_args.kan_hidden2 = args.kan_hidden2
        train_args.kan_width = args.kan_width
        
        # Train model
        train_main(train_args)
        
        # Augmented training
        print(f"\nTraining {model_name.upper()} with augmentation...")
        
        # Update args for augmented training
        train_args.augment = True
        train_args.epochs = args.aug_epochs
        
        # Train model
        train_main(train_args)


def optimize_hyperparameters(data_dir, save_dir, args):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        data_dir (str): Path to the dataset
        save_dir (str): Directory to save optimization results
        args (argparse.Namespace): Command line arguments
        
    Returns:
        optuna.study.Study: Completed Optuna study
    """
    print("\n=== Running Hyperparameter Optimization ===")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Run optimization
    study = run_hyperparameter_optimization(
        data_dir,
        num_trials=args.num_trials,
        num_epochs=args.opt_epochs,
        device=device,
        save_dir=save_dir
    )
    
    # Get best model
    best_model = get_best_model_from_study(study, device=device)
    
    # Save best model
    best_model_path = os.path.join(save_dir, "best_model.pth")
    torch.save(best_model.state_dict(), best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Evaluate best model
    evaluate_model(best_model_path, data_dir)
    
    return study


def evaluate_models(data_dir, save_dir):
    """
    Evaluate trained models.
    
    Args:
        data_dir (str): Path to the dataset
        save_dir (str): Directory containing saved models
    """
    print("\n=== Evaluating Models ===")
    
    # Get all model files
    model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    
    # Evaluate each model
    for model_file in model_files:
        print(f"\nEvaluating {model_file}...")
        model_path = os.path.join(save_dir, model_file)
        evaluate_model(model_path, data_dir)


def run_pipeline(args):
    """
    Run the full training pipeline.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    start_time = time.time()
    
    # Create timestamp for results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.save_dir, f"results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Download and extract dataset
    print("=== Downloading Dataset ===")
    data_dir = download_and_extract_idenprof(args.data_dir)
    
    # Create visualizations
    if args.visualize:
        viz_dir = os.path.join(results_dir, "visualizations")
        create_visualizations(data_dir, viz_dir)
    
    # Run optimization
    if args.optimize:
        opt_dir = os.path.join(results_dir, "optimization")
        study = optimize_hyperparameters(data_dir, opt_dir, args)
    
    # Train models
    if args.train:
        models_dir = os.path.join(results_dir, "models")
        train_models(data_dir, models_dir, args)
    
    # Evaluate models
    if args.evaluate and args.train:
        evaluate_models(data_dir, models_dir)
    
    # Print execution time
    total_time = time.time() - start_time
    print(f"\n=== Pipeline Execution Time: {total_time:.2f}s ===")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the profession classifier training pipeline")
    
    # Pipeline stages
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualizations')
    parser.add_argument('--optimize', action='store_true',
                      help='Run hyperparameter optimization')
    parser.add_argument('--train', action='store_true',
                      help='Train models')
    parser.add_argument('--evaluate', action='store_true',
                      help='Evaluate models')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory to save the dataset')
    parser.add_argument('--save_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--image_size', type=int, default=224,
                      help='Size to resize the images to')
    
    # Model parameters
    parser.add_argument('--models', type=str, 
                      default='vgg,vgg_small,resnet,unet,vit_tiny',
                      help='Comma-separated list of models to train')
    
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
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Batch size for training')
    parser.add_argument('--optimizer', type=str, default='sgd',
                      choices=['sgd', 'adam', 'adamw'],
                      help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=0.05,
                      help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum (for SGD)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                      help='Weight decay')
    parser.add_argument('--epochs', type=int, default=40,
                      help='Number of epochs for standard training')
    parser.add_argument('--aug_epochs', type=int, default=100,
                      help='Number of epochs for augmented training')
    parser.add_argument('--early_stopping', action='store_true',
                      help='Use early stopping')
    parser.add_argument('--patience', type=int, default=5,
                      help='Patience for early stopping')
    
    # Optimization parameters
    parser.add_argument('--num_trials', type=int, default=50,
                      help='Number of optimization trials')
    parser.add_argument('--opt_epochs', type=int, default=20,
                      help='Maximum number of epochs per optimization trial')
    
    # Augmentation parameters
    parser.add_argument('--rotation', type=int, default=30,
                      help='Maximum rotation angle for augmentation')
    parser.add_argument('--hue', type=float, default=0.05,
                      help='Maximum hue jitter factor for augmentation')
    parser.add_argument('--saturation', type=float, default=0.05,
                      help='Maximum saturation jitter factor for augmentation')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                      help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Run at least one stage if none specified
    if not any([args.visualize, args.optimize, args.train, args.evaluate]):
        args.visualize = True
        args.train = True
        args.evaluate = True
    
    run_pipeline(args)
