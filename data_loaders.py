"""
Script for preparing data train, validation and test data loaders
"""
import os
from torchvision import datasets
from torch.utils.data import DataLoader
import augmentation

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    input_size: int,
    train_dir: str,
    val_dir: str,
    test_dir: str,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    # Augmentation function for training and testing data
    train_transform, test_val_transform = augmentation.apply_augmentation(input_size=input_size)

    # Load datasets with ImageFolder
    train_data = datasets.ImageFolder(
        root=train_dir, transform=train_transform, target_transform=None
    )

    val_data = datasets.ImageFolder(
        root=val_dir, transform=test_val_transform, target_transform=None
    )

    test_data = datasets.ImageFolder(
        root=test_dir, transform=test_val_transform, target_transform=None
    )

    # Get class names
    class_names = train_data.classes

    # Print the number of samples in each class for training, validation, and test sets
    print("[INFO] Number of samples in each class for training set:")
    for class_name in class_names:
        class_count = sum(1 for _, label in train_data.samples if train_data.classes[label] == class_name)
        print(f"[INFO] Class {class_name}: {class_count} samples")

    print("\n[INFO] Number of samples in each class for validation set:")
    for class_name in class_names:
        class_count = sum(1 for _, label in val_data.samples if val_data.classes[label] == class_name)
        print(f"[INFO] Class {class_name}: {class_count} samples")

    print("\n[INFO] Number of samples in each class for test set:")
    for class_name in class_names:
        class_count = sum(1 for _, label in test_data.samples if test_data.classes[label] == class_name)
        print(f"[INFO] Class {class_name}: {class_count} samples")

    # Create DataLoaders
    train_dataloader = DataLoader(
        dataset=train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    val_dataloader = DataLoader(
        dataset=val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    test_dataloader = DataLoader(
        dataset=test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader, class_names
