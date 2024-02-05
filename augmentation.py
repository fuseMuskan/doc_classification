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
        transforms.RandomRotation(75),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        transforms.ElasticTransform(alpha=1.0, sigma=20.0),
        transforms.ToTensor(),
    ])

    return train_augmentation, test_val_augmentation

