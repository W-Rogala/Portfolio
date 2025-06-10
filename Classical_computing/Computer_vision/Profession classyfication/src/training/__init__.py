"""
Initialize training package.
"""

from src.training.train import train, evaluate_accuracy
from src.training.evaluate import evaluate_model

__all__ = [
    'train',
    'evaluate_accuracy',
    'evaluate_model'
]