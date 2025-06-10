# Enhanced Profession Classifier
A computer vision project for classifying professionals into 10 different categories using advanced deep learning techniques.

#Overview
This project implements and compares a wide range of CNN and transformer architectures on the "idenprof" dataset which contains images of people from 10 different professions. The main focus is to evaluate different model architectures, the impact of data augmentation, hyperparameter optimization, and Kolmogorov-Arnold Networks (KAN).

# The 10 professions in the dataset are:

Chef,
Doctor,
Engineer,
Farmer,
Firefighter,
Judge,
Mechanic,
Pilot,
Police,
Waiter,
Features,

Multiple Model Architectures:

Traditional CNNs: VGG-11, small VGG-11, ResNet-18
U-Net adapted for classification

Vision Transformers (ViT): Tiny, Small, and Base variants

COCO pre-trained models: ResNet-50, ResNet-101, EfficientNet, Faster R-CNN

# Advanced Techniques:
Data augmentation (rotation, color jittering, flipping)
Hyperparameter optimization with Optuna
Kolmogorov-Arnold Networks (KAN) as alternative to traditional neural networks
Early stopping and model checkpointing
Comprehensive Visualization:
Data exploration and augmentation preview
Model performance comparison
Feature visualization with t-SNE
KAN activation visualization
Attention map visualization for transformers
Installation
bash
# Clone the repository

git clone https://github.com/W-Rogala/Portfolio/edit/main/Classical_computing/Computer_vision/Profession%20classyfication.git

cd enhanced-profession-classifier

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
Quick Start
Data Preparation
bash
# Download and extract the dataset
python -m src.data.download
Training Models
bash
# Train a basic model without augmentation
python -m src.training.train --model vgg --epochs 40

# Train with augmentation
python -m src.training.train --model resnet --augment --epochs 100

# Train a ViT model with KAN layers
python -m src.training.train --model vit_tiny --use_kan --optimizer adam --lr 0.0003 --batch_size 32 --augment
Hyperparameter Optimization
bash
# Run hyperparameter optimization
python -m src.training.optuna_optimization --data_dir data/idenprof --num_trials 50
Complete Pipeline
bash
# Run the complete pipeline
python -m src.run_pipeline --visualize --optimize --train --evaluate
Model Performance
Based on our experiments, here's the approximate performance of different models:

Model	Without Augmentation	With Augmentation	With KAN Layers
VGG-11	~82%	~87%	~88%
Small VGG-11	~77%	~83%	~85%
ResNet-18	~85%	~89%	~90%
U-Net	~79%	~84%	~86%
ViT-Tiny	~81%	~87%	~88%
ViT-Small	~83%	~89%	~91%
COCO-ResNet50	~88%	~92%	~93%
Hyperparameter optimization typically improves performance by an additional 2-5%.

Project Structure
data/: Directory for storing the dataset
models/: Directory for saving trained models
notebooks/: Jupyter notebooks for exploration and visualization
exploration.ipynb: Basic data exploration
vit_and_kan_exploration.ipynb: Vision Transformer and KAN exploration
optimization_results/: Results from hyperparameter optimization
src/: Source code
data/: Data downloading and preprocessing
models/: Model architectures
vgg.py: VGG-11 and small VGG-11 models
resnet.py: ResNet-18 model
unet.py: U-Net adapted for classification
vit.py: Vision Transformer models (Tiny, Small, Base)
coco_models.py: Models pre-trained on COCO dataset
kan.py: Kolmogorov-Arnold Network implementation
training/: Training and evaluation scripts
train.py: Model training with various options
evaluate.py: Model evaluation
optuna_optimization.py: Hyperparameter optimization
utils/: Utility functions
visualization.py: Data visualization
metrics.py: Metrics computation
predict.py: Prediction utilities
model_comparison.py: Model comparison and advanced visualization
run_pipeline.py: End-to-end pipeline
Advanced Usage
Using KAN Layers
KAN (Kolmogorov-Arnold Networks) are a novel neural network architecture based on the Kolmogorov-Arnold representation theorem. They can be applied to any model:

bash
python -m src.training.train --model resnet --use_kan --kan_width 32 --kan_hidden1 128 --kan_hidden2 64
Training Vision Transformers
bash
# Train a tiny ViT
python -m src.training.train --model vit_tiny --optimizer adamw --lr 0.0003 --batch_size 32 --augment

# Train a small ViT
python -m src.training.train --model vit_small --optimizer adamw --lr 0.0001 --batch_size 16 --augment
Using COCO Pre-trained Models
bash
# Train a COCO ResNet-50 model
python -m src.training.train --model coco_resnet50 --optimizer adam --lr 0.0001 --augment
Comparing Models
bash
# Run model comparison
python -m src.utils.model_comparison --models_dir models --compare
Documentation
README.md: This file
PROJECT_GUIDE.md: Detailed guide to the original project
ENHANCED_GUIDE.md: Guide to the enhanced features
Author
Wojciech Rogala

License
This project is licensed under the MIT License - see the LICENSE file for details.

