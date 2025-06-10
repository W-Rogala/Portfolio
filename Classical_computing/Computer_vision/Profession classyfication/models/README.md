Models Directory
This directory contains saved models for the profession classifier project.

Model Types
The following model architectures are implemented:

VGG-11: A version of the Visual Geometry Group network with 11 layers
Small VGG-11: A smaller version of VGG-11 with fewer parameters
ResNet-18: A Residual Network with 18 layers
Training Variants
For each model architecture, there are two training variants:

Standard Training: Trained without data augmentation
Augmented Training: Trained with data augmentation (random rotations, flipping, color jittering)
File Naming Convention
Models are saved with the following naming convention:

{model_name}_{training_type}.pth
Examples:

vgg_standard.pth: VGG-11 model trained without augmentation
vgg_augmented.pth: VGG-11 model trained with augmentation
vgg_small_standard.pth: Small VGG-11 model trained without augmentation
vgg_small_augmented.pth: Small VGG-11 model trained with augmentation
resnet_standard.pth: ResNet-18 model trained without augmentation
resnet_augmented.pth: ResNet-18 model trained with augmentation
Training Models
To train models, use the training script:

bash
python -m src.training.train --model [vgg|vgg_small|resnet] --augment [True|False]
Or use the Makefile:

bash
# Train VGG-11 without augmentation
make train-vgg

# Train VGG-11 with augmentation
make train-vgg-aug

# Train all models
make train
