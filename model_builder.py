"""
Pytorch model code to instantiate different models
"""
from torch import nn
from torchvision import models
import data_loaders
from torch.hub import load_state_dict_from_url
from torchvision.models._api import WeightsEnum


def prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size):
    print("[INFO] Preparing Data Loaders ..")
    (   
        train_dataloader,
        val_dataloader,
        test_dataloader,
        class_names
    ) = data_loaders.create_dataloaders(
        input_size=input_size,
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=batch_size,
    )
    print("[INFO] Data Loaders Prepared.")
    return train_dataloader, val_dataloader, test_dataloader, class_names


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model_name: str,
    num_classes: int,
    feature_extract: bool,
    train_dir,
    test_dir,
    val_dir,
    batch_size,
    use_pretrained=True,
):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """Resnet18"""
        print(f"[INFO] Initializing resnet18 model")

        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
        ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)
    

    elif model_name == "resnet50":
      """Resnet 50"""
      print("[INFO] Initializing resnet50 model")
      model_ft = models.resnet50(pretrained=use_pretrained)
      set_parameter_requires_grad(model_ft, feature_extract)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, num_classes)
      input_size = 224

      (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        class_names,
      ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)

    elif model_name == "alexnet":
        """Alexnet"""
        print(f"[INFO] Initializing alexnet model")

        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
        ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)

    elif model_name == "vgg":
        """VGG11_bn"""
        print(f"[INFO] Initializing vgg11 model")

        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
        ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)

    elif model_name == "squeezenet":
        """Squeezenet"""
        print(f"[INFO] Initializing squeezenet model")

        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
        ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)

    elif model_name == "densenet":
        """Densenet"""
        print(f"[INFO] Initializing densenet model")

        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

        (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
        ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)

    elif model_name == "efficientnet":
      """Efficient Net B0"""

      print(f"[INFO] Initializing efficient net b0 model")

      def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)
    
      WeightsEnum.get_state_dict = get_state_dict
      
      model_ft = models.efficientnet_b0(weights="DEFAULT")
      set_parameter_requires_grad(model_ft,feature_extract)
      model_ft.classifier[1] = nn.Linear(in_features=1280, out_features=4)
      input_size = 224
      (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            class_names,
      ) = prepare_dataloaders(input_size, train_dir, test_dir, val_dir, batch_size)


    else:
        print("Invalid model name, exiting...")
        exit()
    
    print(f"[INFO] {model_name} Successfully Initialized.")


    return (
        model_ft,
        input_size,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        class_names,
    )
