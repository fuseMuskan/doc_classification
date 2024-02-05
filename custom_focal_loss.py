"""
Script for defining a Focal Loss class and related functions for class weights calculation.

This script provides a FocalLoss class, a function to calculate class weights based on the training dataset,
and a function to create a Focal Loss criterion using the calculated class weights.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_class_weights(
    train_dataloader: torch.utils.data.DataLoader, class_names: list
) -> torch.FloatTensor:
    """
    Calculates the class weights based on the samples present in the train dataset.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Train dataloader.
        class_names (list): List of available classes.

    Returns:
        torch.FloatTensor: Returns a tensor of class weights sequentially based on the class_names.
    """
    class_counts = {class_name: 0 for class_name in class_names}
    for _, targets in train_dataloader:
        for target in targets:
            class_name = class_names[target]
            class_counts[class_name] += 1

    # Calculate class weights
    total_samples = sum(class_counts.values())
    class_weights = [
        total_samples / count for class_name, count in class_counts.items()
    ]
    class_weights = torch.FloatTensor(class_weights)
    return class_weights


class FocalLoss(nn.Module):
    def __init__(self, alpha: list, gamma: int or float = 2):
        """
        Initializes alpha and gamma for the focal loss function.

        Args:
            alpha (list): Class weights.
            gamma (int or float, optional): Value of gamma to be used in the loss function. Defaults to 2.

        Returns:
            None
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Input from the forward pass of the model.
            targets (torch.Tensor): Corresponding targets of the inputs.

        Returns:
            torch.Tensor: Calculated loss after the forward pass.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = (self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss).mean()
        return loss


def create_focal_loss_criterion(
    train_dataloader: torch.utils.data.DataLoader,
    class_names: list,
    gamma: int or float = 2,
) -> FocalLoss:
    """
    Uses the calculate_class_weights function and FocalLoss class to return the
    loss function based on the class weights.

    Args:
        train_dataloader (torch.utils.data.DataLoader): Train dataloader.
        class_names (list): List of class names.
        gamma (int or float): Value of gamma to be used in the loss function. Defaults to 2.

    Returns:
        FocalLoss: Returns an instance of the FocalLoss class.
    """
    class_weights = calculate_class_weights(train_dataloader, class_names)
    criterion = FocalLoss(alpha=class_weights, gamma=gamma)
    return criterion
