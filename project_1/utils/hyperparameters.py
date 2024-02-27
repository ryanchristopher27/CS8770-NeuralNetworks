# Imports
import torch.optim as optim
import torch.nn as nn


def get_optimizer(
    optimizer_name: str,
    model,
    learning_rate: float,
    momentum: float,
):
    options = ['SGD', 'ADAM']
    if optimizer_name not in options:
        raise ValueError(f'Invalid Optimizer: {optimizer_name}')
    
    if optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    
    elif optimizer_name == "ADAM":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return optimizer


def get_criterion(
    criterion_name: str,
):
    options = ['CrossEntropy']
    if criterion_name not in options:
        raise ValueError(f'Invalid Criterion: {criterion_name}')
    
    if criterion_name == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()

    return criterion