"""
Initialize models package with all available architectures.
"""

from src.models.vgg import vgg_11, vgg_11_small
from src.models.resnet import resnet18
from src.models.unet import unet
from src.models.vit import vit_tiny, vit_small, vit_base
from src.models.coco_models import coco_classifier, faster_rcnn_classifier
from src.models.kan import add_kan_layer, KANLayer, KANNetwork, KANWrapper

__all__ = [
    # Base models
    'vgg_11',
    'vgg_11_small',
    'resnet18',
    'unet',
    'vit_tiny',
    'vit_small',
    'vit_base',
    'coco_classifier',
    'faster_rcnn_classifier',
    
    # KAN layers
    'add_kan_layer',
    'KANLayer',
    'KANNetwork',
    'KANWrapper'
]