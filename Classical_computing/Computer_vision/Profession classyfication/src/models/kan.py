"""
Kolmogorov-Arnold Network (KAN) layer implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network Layer.
    
    Implementation based on the paper "Kolmogorov-Arnold Networks: A Mathematical Framework for Transparent Deep Learning"
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        width (int): Width of the KAN layer (number of inner functions)
        init_scale (float): Scale of initialization
    """
    def __init__(self, in_features, out_features, width=16, init_scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.width = width
        
        # Projection matrices
        self.W1 = nn.Parameter(init_scale * torch.randn(in_features, width))
        self.W2 = nn.Parameter(init_scale * torch.randn(width, out_features))
        
        # Bias terms
        self.b1 = nn.Parameter(torch.zeros(width))
        self.b2 = nn.Parameter(torch.zeros(out_features))
        
        # Parameters for the univariate functions
        self.univariate_weights = nn.Parameter(init_scale * torch.randn(width, 8))
        self.univariate_bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        # Project inputs into the univariate function space
        projections = F.linear(x, self.W1, self.b1)
        
        # Apply univariate functions (using Chebyshev polynomials)
        activations = self._chebyshev_activation(projections)
        
        # Project back to output space
        return F.linear(activations, self.W2, self.b2)
    
    def _chebyshev_activation(self, x):
        """
        Apply Chebyshev polynomial approximation as univariate functions.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, width]
            
        Returns:
            torch.Tensor: Activated tensor [batch_size, width]
        """
        # Scale x to the range [-1, 1] for Chebyshev polynomials
        x_scaled = torch.clamp(x, -3, 3) / 3.0
        
        # Compute Chebyshev polynomials T_0(x) through T_7(x)
        t0 = torch.ones_like(x_scaled)
        t1 = x_scaled
        t2 = 2 * x_scaled * t1 - t0  # 2x² - 1
        t3 = 2 * x_scaled * t2 - t1  # 4x³ - 3x
        t4 = 2 * x_scaled * t3 - t2  # 8x⁴ - 8x² + 1
        t5 = 2 * x_scaled * t4 - t3  # 16x⁵ - 20x³ + 5x
        t6 = 2 * x_scaled * t5 - t4  # 32x⁶ - 48x⁴ + 18x² - 1
        t7 = 2 * x_scaled * t6 - t5  # 64x⁷ - 112x⁵ + 56x³ - 7x
        
        # Stack Chebyshev polynomials
        cheb = torch.stack([t0, t1, t2, t3, t4, t5, t6, t7], dim=-1)
        
        # Weight combination of Chebyshev polynomials
        output = torch.matmul(cheb, self.univariate_weights.t()).diagonal(dim1=1, dim2=2)
        
        # Add bias
        output = output + self.univariate_bias
        
        return output


class KANNetwork(nn.Module):
    """
    Network with KAN layers.
    
    Args:
        in_features (int): Number of input features
        hidden_sizes (list): List of hidden layer sizes
        out_features (int): Number of output features
        width (int): Width of KAN layers
    """
    def __init__(self, in_features, hidden_sizes, out_features, width=16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Build network
        sizes = [in_features] + hidden_sizes + [out_features]
        layers = []
        
        for i in range(len(sizes) - 1):
            layers.append(KANLayer(sizes[i], sizes[i+1], width=width))
            
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply ReLU activation to all but the last layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class KANWrapper(nn.Module):
    """
    Wrapper to add KAN layers to standard CNN models.
    
    Args:
        base_model (nn.Module): Base CNN model
        kan_hidden_sizes (list): List of hidden layer sizes for KAN network
        num_classes (int): Number of output classes
        kan_width (int): Width of KAN layers
    """
    def __init__(self, base_model, kan_hidden_sizes=[128, 64], num_classes=10, kan_width=16):
        super().__init__()
        self.base_model = base_model
        
        # Extract features from CNN (remove the last fully connected layer)
        if hasattr(base_model, 'fc'):
            in_features = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif hasattr(base_model, 'classifier') and isinstance(base_model.classifier, nn.Sequential):
            # For VGG-like models
            in_features = base_model.classifier[-1].in_features
            base_model.classifier[-1] = nn.Identity()
        elif hasattr(base_model, 'head'):
            # For ViT models
            in_features = base_model.head.in_features
            base_model.head = nn.Identity()
        else:
            raise ValueError("Cannot determine feature size of the base model")
        
        # KAN classifier
        self.kan_classifier = KANNetwork(
            in_features, kan_hidden_sizes, num_classes, width=kan_width
        )
    
    def forward(self, x):
        # Extract features from base model
        features = self.base_model(x)
        # Apply KAN classifier
        return self.kan_classifier(features)


def add_kan_layer(model, kan_hidden_sizes=[128, 64], num_classes=10, kan_width=16):
    """
    Add KAN layers to a model.
    
    Args:
        model (nn.Module): Base model
        kan_hidden_sizes (list): List of hidden layer sizes for KAN network
        num_classes (int): Number of output classes
        kan_width (int): Width of KAN layers
        
    Returns:
        nn.Module: Model with KAN layers
    """
    return KANWrapper(model, kan_hidden_sizes, num_classes, kan_width)


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test KAN layer
    x = torch.randn(10, 5)
    kan_layer = KANLayer(5, 3, width=16)
    output = kan_layer(x)
    print(f"KAN Layer output shape: {output.shape}")
    
    # Test KAN network
    kan_net = KANNetwork(5, [10, 8], 3, width=16)
    output = kan_net(x)
    print(f"KAN Network output shape: {output.shape}")
    
    # Test KAN wrapper with a dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.fc = nn.Linear(16 * 56 * 56, 10)
        
        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    dummy_model = DummyModel()
    kan_wrapped = add_kan_layer(dummy_model)
    
    # Test with dummy input
    x = torch.randn(2, 3, 224, 224)
    try:
        output = kan_wrapped(x)
        print(f"KAN Wrapped Model output shape: {output.shape}")
        print(f"KAN Wrapped Model parameter count: {count_parameters(kan_wrapped):,}")
    except Exception as e:
        print(f"Error in KAN wrapped model: {e}")
