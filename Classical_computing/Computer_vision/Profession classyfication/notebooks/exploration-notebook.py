{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Profession Classifier - Data Exploration\n",
    "\n",
    "This notebook explores the idenprof dataset and visualizes the effects of data augmentation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "import os\n",
    "import sys\n",
    "import random\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "import torch\n",
    "from torchvision import transforms, datasets\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "# Add project root to path\n",
    "sys.path.append('..')\n",
    "\n",
    "from src.data.download import download_and_extract_idenprof\n",
    "from src.data.preprocess import get_data_loaders, get_augmented_data_loaders, get_class_names\n",
    "from src.utils.visualization import visualize_sample, visualize_augmentation, visualize_class_distribution"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Download and Extract Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Download and extract dataset\n",
    "data_dir = download_and_extract_idenprof(data_dir='../data')\n",
    "print(f\"Dataset directory: {data_dir}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Explore Dataset Structure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Get class names\n",
    "class_names = get_class_names(data_dir)\n",
    "print(f\"Class names: {class_names}\")\n",
    "\n",
    "# Print dataset statistics\n",
    "train_dir = os.path.join(data_dir, 'train')\n",
    "test_dir = os.path.join(data_dir, 'test')\n",
    "\n",
    "print(\"\\nSamples per class:\")\n",
    "print(\"-\" * 40)\n",
    "print(f\"{'Class':<15} {'Train':<10} {'Test':<10} {'Total':<10}\")\n",
    "print(\"-\" * 40)\n",
    "\n",
    "total_train = 0\n",
    "total_test = 0\n",
    "\n",
    "for class_idx, class_name in class_names.items():\n",
    "    train_class_dir = os.path.join(train_dir, class_name)\n",
    "    test_class_dir = os.path.join(test_dir, class_name)\n",
    "    \n",
    "    train_count = len(os.listdir(train_class_dir))\n",
    "    test_count = len(os.listdir(test_class_dir))\n",
    "    total = train_count + test_count\n",
    "    \n",
    "    total_train += train_count\n",
    "    total_test += test_count\n",
    "    \n",
    "    print(f\"{class_name:<15} {train_count:<10} {test_count:<10} {total:<10}\")\n",
    "\n",
    "print(\"-\" * 40)\n",
    "print(f\"{'Total':<15} {total_train:<10} {total_test:<10} {total_train + total_test:<10}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Visualize Class Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize class distribution\n",
    "visualize_class_distribution(data_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Sample Images"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize random samples\n",
    "visualize_sample(data_dir, num_samples=8, seed=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Data Augmentation Effects"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Visualize augmentation effects\n",
    "visualize_augmentation(\n",
    "    data_dir,\n",
    "    rotation=30,\n",
    "    hue=0.5,     # Exaggerated for visualization\n",
    "    saturation=0.5,  # Exaggerated for visualization\n",
    "    seed=42\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Multiple Augmentation Examples"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Pick a single image and show multiple augmentations\n",
    "def show_multiple_augmentations(img_path, n_augmentations=5, seed=None):\n",
    "    # Set seed for reproducibility\n",
    "    if seed is not None:\n",
    "        random.seed(seed)\n",
    "        torch.manual_seed(seed)\n",
    "    \n",
    "    # Define transform\n",
    "    transform_augmented = transforms.Compose([\n",
    "        transforms.ColorJitter(hue=0.05, saturation=0.05),\n",
    "        transforms.RandomRotation(30, expand=True),\n",
    "        transforms.RandomVerticalFlip(),\n",
    "        transforms.RandomHorizontalFlip(),\n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.ToTensor()\n",
    "    ])\n",
    "    \n",
    "    # Load image\n",
    "    img = Image.open(img_path).convert('RGB')\n",
    "    \n",
    "    # Original image\n",
    "    transform_original = transforms.Compose([\n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.ToTensor()\n",
    "    ])\n",
    "    \n",
    "    img_original = transform_original(img)\n",
    "    \n",
    "    # Plot\n",
    "    fig, axes = plt.subplots(1, n_augmentations+1, figsize=(15, 3))\n",
    "    \n",
    "    # Display original\n",
    "    axes[0].imshow(img_original.permute(1, 2, 0))\n",
    "    axes[0].set_title(\"Original\")\n",
    "    axes[0].axis('off')\n",
    "    \n",
    "    # Display augmentations\n",
    "    for i in range(n_augmentations):\n",
    "        img_augmented = transform_augmented(img)\n",
    "        axes[i+1].imshow(img_augmented.permute(1, 2, 0))\n",
    "        axes[i+1].set_title(f\"Augmentation {i+1}\")\n",
    "        axes[i+1].axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "# Find a sample image\n",
    "sample_profession = \"chef\"\n",
    "sample_dir = os.path.join(data_dir, 'train', sample_profession)\n",
    "sample_image = os.path.join(sample_dir, os.listdir(sample_dir)[0])\n",
    "\n",
    "# Show multiple augmentations\n",
    "show_multiple_augmentations(sample_image, n_augmentations=5, seed=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Data Loaders Testing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Test data loaders\n",
    "batch_size = 4\n",
    "\n",
    "# Standard data loader\n",
    "train_loader, test_loader = get_data_loaders(data_dir, batch_size=batch_size)\n",
    "\n",
    "# Get a batch of images\n",
    "dataiter = iter(train_loader)\n",
    "images, labels = next(dataiter)\n",
    "\n",
    "# Show images\n",
    "plt.figure(figsize=(10, 4))\n",
    "for i in range(batch_size):\n",
    "    plt.subplot(1, batch_size, i+1)\n",
    "    # Un-normalize\n",
    "    img = images[i] / 2 + 0.5\n",
    "    plt.imshow(img.permute(1, 2, 0))\n",
    "    plt.title(class_names[labels[i].item()])\n",
    "    plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Augmented data loader\n",
    "aug_train_loader, aug_test_loader = get_augmented_data_loaders(\n",
    "    data_dir, \n",
    "    batch_size=batch_size,\n",
    "    rotation=30,\n",
    "    hue=0.05,\n",
    "    saturation=0.05\n",
    ")\n",
    "\n",
    "# Get a batch of images\n",
    "aug_dataiter = iter(aug_train_loader)\n",
    "aug_images, aug_labels = next(aug_dataiter)\n",
    "\n",
    "# Show images\n",
    "plt.figure(figsize=(10, 4))\n",
    "for i in range(batch_size):\n",
    "    plt.subplot(1, batch_size, i+1)\n",
    "    # Un-normalize\n",
    "    img = aug_images[i] / 2 + 0.5\n",
    "    plt.imshow(img.permute(1, 2, 0))\n",
    "    plt.title(class_names[aug_labels[i].item()])\n",
    "    plt.axis('off')\n",
    "plt.tight_layout()\n",
    "plt.suptitle(\"Augmented Batch\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Image Size Analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "source": [
    "# Analyze image sizes in the dataset\n",
    "def analyze_image_sizes(data_dir):\n",
    "    # Get all image paths\n",
    "    train_dir = os.path.join(data_dir, 'train')\n",
    "    image_paths = []\n",
    "    \n",
    "    for profession in os.listdir(train_dir):\n",
    "        profession_dir = os.path.join(train_dir, profession)\n",
    "        for image_file in os.listdir(profession_dir):\n",
    "            image_paths.append(os.path.join(profession_dir, image_file))\n",
    "    \n",
    "    # Sample 100 images\n",
    "    sample_paths = random.sample(image_paths, min(100, len(image_paths)))\n",
    "    \n",
    "    # Analyze sizes\n",
    "    widths = []\n",
    "    heights = []\n",
    "    aspect_ratios = []\n",
    "    \n",
    "    for path in sample_paths:\n",
    "        img = Image.open(path)\n",
    "        width, height = img.size\n",
    "        aspect_ratio = width / height\n",
    "        \n",
    "        widths.append(width)\n",
    "        heights.append(height)\n",
    "        aspect_ratios.append(aspect_ratio)\n",
    "    \n",
    "    # Plot distributions\n",
    "    fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
    "    \n",
    "    # Width distribution\n",
    "    axes[0].hist(widths, bins=20)\n",
    "    axes[0].set_title('Width Distribution')\n",
    "    axes[0].set_xlabel('Width (pixels)')\n",
    "    axes[0].set_ylabel('Count')\n",
    "    \n",
    "    # Height distribution\n",
    "    axes[1].hist(heights, bins=20)\n",
    "    axes[1].set_title('Height Distribution')\n",
    "    axes[1].set_xlabel('Height (pixels)')\n",
    "    \n",
    "    # Aspect ratio distribution\n",
    "    axes[2].hist(aspect_ratios, bins=20)\n",
    "    axes[2].set_title('Aspect Ratio Distribution')\n",
    "    axes[2].set_xlabel('Aspect Ratio (width/height)')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    # Print statistics\n",
    "    print(f\"Width: min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.2f}\")\n",
    "    print(f\"Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.2f}\")\n",
    "    print(f\"Aspect Ratio: min={min(aspect_ratios):.2f}, max={max(aspect_ratios):.2f}, mean={np.mean(aspect_ratios):.2f}\")\n",
    "\n",
    "# Analyze image sizes\n",
    "analyze_image_sizes(data_dir)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this notebook, we have explored the idenprof dataset and visualized the effects of data augmentation. The dataset contains images of people from 10 different professions, and we have observed that data augmentation can significantly increase the diversity of the training data.\n",
    "\n",
    "Key findings:\n",
    "- The dataset is fairly balanced across professions\n",
    "- Images have varying sizes and aspect ratios, which we normalize to 224x224 pixels\n",
    "- Data augmentation techniques like rotation, flipping, and color jittering can create diverse training samples\n",
    "\n",
    "Next steps:\n",
    "- Train models with and without data augmentation\n",
    "- Compare model performance on the test set\n",
    "- Analyze the impact of different augmentation techniques"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}