"""
Script for calculating confusion matrix, plotting it, and computing classification report.

This script provides functions to calculate the confusion matrix, plot it using seaborn and matplotlib,
and compute a classification report using scikit-learn. The results can be saved in log files.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from PIL import Image
from torcheval.metrics import MulticlassConfusionMatrix
from sklearn.metrics import classification_report
from datetime import datetime

class_names = ["citizenship", "license", "others", "passport"]


def calculate_confusion_matrix(
    input: torch.Tensor or torch.cuda.FloatTensor, target: torch.Tensor or torch.cuda.FloatTensor
) -> torch.Tensor or torch.cuda.FloatTensor:
    """
    Calculate the confusion matrix for the given predictions and targets.

    Args:
        input (torch.Tensor): Model predictions.
        target (torch.Tensor): True labels.

    Returns:
        torch.Tensor: Confusion matrix.
    """
    metric = MulticlassConfusionMatrix(4)
    metric.update(input, target)
    conf_matrix = metric.compute()
    return conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names: list) -> plt.Figure:
    """
    Plot and visualize the confusion matrix.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        class_names (list): List of class names.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(len(class_names), len(class_names)))
    sns.set(font_scale=1.2)
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    return fig


def save_confusion_matrix(conf_matrix: torch.Tensor) -> plt.Figure:
    """
    Save and return the confusion matrix visualization as a figure.

    Args:
        conf_matrix (torch.Tensor): Confusion matrix.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    class_names = ["citizenship", "license", "others", "passport"]

    fig = plot_confusion_matrix(
        conf_matrix=conf_matrix.numpy(), class_names=class_names
    )
    return fig


def compute_classification_report(
    input: torch.Tensor, target: torch.Tensor, model_name: str
) -> None:
    """
    Compute and save the classification report.

    Args:
        input (torch.Tensor): Model predictions.
        target (torch.Tensor): True labels.
        model_name (str): Name of the model.
    """
    # Convert tensors to NumPy arrays
    y_true = target.numpy()
    y_pred = input.numpy()

    # Compute classification report
    report = classification_report(y_true, y_pred, target_names=class_names)

    # Generating Classification Report
    print("----Generating Classification Report---------")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = f"logs/classification_report_{model_name}_{timestamp}.txt"

    # Print and save the report to the log file
    with open(log_file, "w") as f:
        f.write(
            f"-------------------{model_name} Classification Report-------------------\n"
        )
        f.write(report)
        f.write("-" * 80)

    print("Classification report saved in:", log_file)
