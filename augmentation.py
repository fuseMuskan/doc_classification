"""
Script for applying necessary transformation on the images as per the model requirement
"""
from torchvision import transforms


def apply_augmentation(input_size: int):
    # Transformation for Test and Validation Set
    test_val_augmentation = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    # Transformation for Training set
    train_augmentation = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    return train_augmentation, test_val_augmentation

