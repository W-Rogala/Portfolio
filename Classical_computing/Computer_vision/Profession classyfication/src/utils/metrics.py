"""
Metrics module for the profession classifier project.
"""

import torch


class Accumulator:
    """
    For accumulating sums over n variables.
    """
    def __init__(self, n):
        """
        Initialize an accumulator.
        
        Args:
            n (int): Number of variables to accumulate
        """
        self.data = [0.0] * n
    
    def add(self, *args):
        """
        Add values to the accumulator.
        
        Args:
            *args: Values to add
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        """
        Reset the accumulator to zeros.
        """
        self.data = [0.0] * len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a value from the accumulator.
        
        Args:
            idx (int): Index
            
        Returns:
            float: Value at index
        """
        return self.data[idx]


def accuracy(y_hat, y):
    """
    Compute the accuracy for a given prediction and ground truth.
    
    Args:
        y_hat (torch.Tensor): Prediction
        y (torch.Tensor): Ground truth
        
    Returns:
        float: Accuracy
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # For multi-class classification
        y_hat = y_hat.argmax(axis=1)
    
    correct = (y_hat.type(y.dtype) == y).sum().item()
    return float(correct)


def evaluate_accuracy(net, data_iter, device=None):
    """
    Evaluate the accuracy of a model on a dataset.
    
    Args:
        net (torch.nn.Module): Model
        data_iter (iterable): Data iterator
        device (torch.device): Device
        
    Returns:
        float: Accuracy
    """
    # Set evaluation mode
    if isinstance(net, torch.nn.Module):
        net.eval()
        
    # Get device
    if not device:
        device = next(iter(net.parameters())).device
    
    # Metrics
    metric = Accumulator(2)  # Number of correct predictions, number of predictions
    
    # Evaluation loop
    with torch.no_grad():
        for X, y in data_iter:
            # Move data to device
            X, y = X.to(device), y.to(device)
            
            # Make prediction
            y_hat = net(X)
            
            # Update metrics
            metric.add(accuracy(y_hat, y), len(y))
    
    # Return accuracy
    return metric[0] / metric[1]


def predict_class(net, X, device=None):
    """
    Predict the class of an input.
    
    Args:
        net (torch.nn.Module): Model
        X (torch.Tensor): Input
        device (torch.device): Device
        
    Returns:
        int: Predicted class
    """
    # Set evaluation mode
    if isinstance(net, torch.nn.Module):
        net.eval()
        
    # Get device
    if not device:
        device = next(iter(net.parameters())).device
    
    # Move input to device
    X = X.to(device)
    
    # Make prediction
    with torch.no_grad():
        y_hat = net(X)
        return y_hat.argmax(dim=1).item()


if __name__ == "__main__":
    # Test Accumulator
    acc = Accumulator(2)
    acc.add(1, 2)
    acc.add(3, 4)
    print(f'Accumulator: {acc.data}')
    assert acc[0] == 4 and acc[1] == 6, "Accumulator test failed"
    
    # Test accuracy function
    y_hat = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
    y = torch.tensor([1, 0])
    acc = accuracy(y_hat, y) / len(y)
    print(f'Accuracy: {acc:.4f}')
    assert abs(acc - 0.5) < 1e-6, "Accuracy test failed"
    
    print("All tests passed!")