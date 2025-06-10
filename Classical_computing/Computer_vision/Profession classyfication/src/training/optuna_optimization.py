"""
Hyperparameter optimization module using Optuna.
"""

import os
import torch
import optuna
from optuna.trial import Trial
import logging
import numpy as np
from functools import partial

from src.data.preprocess import get_data_loaders, get_augmented_data_loaders
from src.models.vgg import vgg_11, vgg_11_small
from src.models.resnet import resnet18
from src.models.unet import unet
from src.models.vit import vit_tiny, vit_small, vit_base
from src.models.coco_models import coco_classifier, faster_rcnn_classifier
from src.models.kan import add_kan_layer
from src.training.train import train
from src.utils.metrics import evaluate_accuracy


# Configure logging
logger = logging.getLogger("optuna_optimization")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_model_for_trial(trial: Trial, num_classes=10):
    """
    Get a model for the trial with optimized architecture parameters.
    
    Args:
        trial (optuna.trial.Trial): Optuna trial
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: Model for the trial
    """
    # Select model architecture
    model_name = trial.suggest_categorical(
        "model", [
            "vgg_11", "vgg_11_small", "resnet18", "unet", 
            "vit_tiny", "vit_small", "vit_base",
            "coco_resnet50", "coco_resnet101", "coco_efficientnet", "faster_rcnn"
        ]
    )
    
    # Indicate which model is being used in the current trial
    logger.info(f"Selected model architecture: {model_name}")
    
    # Select if KAN should be applied
    use_kan = trial.suggest_categorical("use_kan", [True, False])
    
    # Create base model
    if model_name == "vgg_11":
        model = vgg_11(num_classes=num_classes)
    elif model_name == "vgg_11_small":
        ratio = trial.suggest_int("vgg_ratio", 2, 8)
        model = vgg_11_small(ratio=ratio, num_classes=num_classes)
    elif model_name == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif model_name == "unet":
        features = trial.suggest_categorical("unet_features", [32, 48, 64])
        bilinear = trial.suggest_categorical("unet_bilinear", [True, False])
        model = unet(n_classes=num_classes, features=features, bilinear=bilinear)
    elif model_name == "vit_tiny":
        model = vit_tiny(num_classes=num_classes)
    elif model_name == "vit_small":
        model = vit_small(num_classes=num_classes)
    elif model_name == "vit_base":
        model = vit_base(num_classes=num_classes)
    elif model_name == "coco_resnet50":
        model = coco_classifier(backbone_type='resnet50', num_classes=num_classes)
    elif model_name == "coco_resnet101":
        model = coco_classifier(backbone_type='resnet101', num_classes=num_classes)
    elif model_name == "coco_efficientnet":
        model = coco_classifier(backbone_type='efficientnet', num_classes=num_classes)
    elif model_name == "faster_rcnn":
        model = faster_rcnn_classifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Apply KAN if selected
    if use_kan:
        kan_width = trial.suggest_categorical("kan_width", [8, 16, 32])
        kan_hidden_sizes = [
            trial.suggest_categorical("kan_hidden_1", [64, 128, 256]),
            trial.suggest_categorical("kan_hidden_2", [32, 64, 128])
        ]
        
        logger.info(f"Applying KAN layer with width={kan_width}, hidden_sizes={kan_hidden_sizes}")
        model = add_kan_layer(
            model, 
            kan_hidden_sizes=kan_hidden_sizes, 
            num_classes=num_classes,
            kan_width=kan_width
        )
    
    return model


def objective(trial: Trial, data_dir, device, num_epochs=20, num_classes=10, save_dir=None):
    """
    Optuna optimization objective function.
    
    Args:
        trial (optuna.trial.Trial): Optuna trial
        data_dir (str): Directory containing the dataset
        device (torch.device): Device to train on
        num_epochs (int): Maximum number of epochs to train
        num_classes (int): Number of output classes
        save_dir (str): Directory to save models
        
    Returns:
        float: Validation accuracy
    """
    # Hyperparameters to optimize
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    use_augmentation = trial.suggest_categorical("use_augmentation", [True, False])
    
    logger.info(f"Trial {trial.number}: batch_size={batch_size}, lr={learning_rate}, "
               f"weight_decay={weight_decay}, use_augmentation={use_augmentation}")
    
    # Data augmentation parameters if enabled
    if use_augmentation:
        rotation = trial.suggest_int("rotation", 10, 45)
        hue = trial.suggest_float("hue", 0.01, 0.1)
        saturation = trial.suggest_float("saturation", 0.01, 0.1)
        
        logger.info(f"Augmentation parameters: rotation={rotation}, hue={hue}, saturation={saturation}")
        
        # Load data with augmentation
        train_loader, test_loader = get_augmented_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            image_size=224,
            rotation=rotation,
            hue=hue,
            saturation=saturation
        )
    else:
        # Load data without augmentation
        train_loader, test_loader = get_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            image_size=224
        )
    
    # Get model for the trial
    model = get_model_for_trial(trial, num_classes=num_classes)
    model.to(device)
    
    # Set optimizer
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW"])
    
    if optimizer_name == "SGD":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    # Set loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Add pruning callback
    pruning_callback = optuna.integration.PyTorchLightningPruningCallback(trial, monitor="val_acc")
    
    # Save path for the model if save_dir is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"trial_{trial.number}.pth")
    else:
        save_path = None
    
    try:
        # Train model
        history, best_acc = train(
            model, 
            train_loader, 
            test_loader, 
            loss_fn, 
            optimizer, 
            num_epochs, 
            device,
            save_path,
            early_stopping=True,
            patience=5,
            trial=trial  # Pass trial for pruning
        )
        
        # Report best accuracy
        logger.info(f"Trial {trial.number} finished with best accuracy: {best_acc:.4f}")
        
        return best_acc
    
    except optuna.exceptions.TrialPruned:
        # Trial was pruned
        logger.info(f"Trial {trial.number} was pruned")
        raise
    
    except Exception as e:
        # Log error and return poor accuracy
        logger.error(f"Error in trial {trial.number}: {e}")
        return 0.0


def run_hyperparameter_optimization(data_dir, num_trials=50, num_epochs=20, 
                                  num_classes=10, device=None, save_dir=None):
    """
    Run hyperparameter optimization using Optuna.
    
    Args:
        data_dir (str): Directory containing the dataset
        num_trials (int): Number of optimization trials
        num_epochs (int): Maximum number of epochs to train per trial
        num_classes (int): Number of output classes
        device (torch.device): Device to train on
        save_dir (str): Directory to save models and study
        
    Returns:
        optuna.study.Study: Completed Optuna study
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create save directory for the study
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        study_path = os.path.join(save_dir, "optuna_study.db")
        study = optuna.create_study(
            study_name="profession_classifier_optimization",
            direction="maximize",
            storage=f"sqlite:///{study_path}",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )
    else:
        study = optuna.create_study(
            study_name="profession_classifier_optimization",
            direction="maximize",
            pruner=optuna.pruners.MedianPruner()
        )
    
    logger.info(f"Starting hyperparameter optimization with {num_trials} trials")
    logger.info(f"Using device: {device}")
    
    # Create objective function with fixed parameters
    objective_func = partial(
        objective, 
        data_dir=data_dir, 
        device=device, 
        num_epochs=num_epochs, 
        num_classes=num_classes,
        save_dir=save_dir
    )
    
    # Run optimization
    study.optimize(objective_func, n_trials=num_trials, timeout=None)
    
    # Log results
    logger.info("Hyperparameter optimization finished")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best accuracy: {study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"    {key}: {value}")
    
    # Save study information if save_dir is provided
    if save_dir:
        # Save best trial parameters
        best_params_path = os.path.join(save_dir, "best_params.txt")
        with open(best_params_path, 'w') as f:
            f.write(f"Best trial: {study.best_trial.number}\n")
            f.write(f"Best accuracy: {study.best_value:.4f}\n")
            f.write("Best hyperparameters:\n")
            for key, value in study.best_params.items():
                f.write(f"    {key}: {value}\n")
    
    return study


def get_best_model_from_study(study, num_classes=10, device=None):
    """
    Create a model with the best hyperparameters from a completed study.
    
    Args:
        study (optuna.study.Study): Completed Optuna study
        num_classes (int): Number of output classes
        device (torch.device): Device to load the model on
        
    Returns:
        torch.nn.Module: Best model from the study
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get best trial
    best_trial = study.best_trial
    
    # Create model with best hyperparameters
    model = get_model_for_trial(best_trial, num_classes=num_classes)
    model.to(device)
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/idenprof",
                      help="Directory containing the dataset")
    
    # Optimization parameters
    parser.add_argument("--num_trials", type=int, default=50,
                      help="Number of optimization trials")
    parser.add_argument("--num_epochs", type=int, default=20,
                      help="Maximum number of epochs per trial")
    
    # Save parameters
    parser.add_argument("--save_dir", type=str, default="optimization_results",
                      help="Directory to save models and study")
    
    # Device parameters
    parser.add_argument("--no_cuda", action="store_true",
                      help="Disable CUDA")
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    
    # Run optimization
    study = run_hyperparameter_optimization(
        args.data_dir,
        num_trials=args.num_trials,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    # Get best model
    best_model = get_best_model_from_study(study, device=device)
    
    # Save best model
    if args.save_dir:
        best_model_path = os.path.join(args.save_dir, "best_model.pth")
        torch.save(best_model.state_dict(), best_model_path)
        print(f"Best model saved to {best_model_path}")
