## Data Collection
The documents (citizenship, license, passport) were scraped from Google Images.



## Getting Started

### Installing dependencies and packages
```
pip install -r requirements.txt
```

## Data Augmentation
Since the scraped documents were very less, different augmentation techniques were applied to increase the training set.
The applied augmentation techniques are as follows:

* Translate
* Scale
* Flip
* Brightness
* Rotate
* Shear
* Zoom
* Crop
* Erasekv
* Elastic

### Usage

You can apply transformation to your own datasets using following command

* data_dir = path to the training dataset
* num_aug_images = no of transformed images to generate for each image
* output_dir = path to the save transformed images

```
python data_transformation.py --data_dir=./path/to/your/train_directory --num_aug_images=5 --output_dir=./path/to/save/output
```

## Pre-trained Models
Following pre-trained models were used for feature extraction
* VGG19
* ResNet18
* ResNet50
* Alexnet
* VGG11
* Squeezenet
* DenseNet
* YOLOv8
* EfficientNet0

### Usage

* MODEL = name of the model to choose ["resnet18", "resnet50", "alexnet", "vgg", "squeezenet", "densenet", "efficientnet", "yolo"]
* EPOCHS = num of epochs
* BATCH_SIZE = batch size
* DATA_DIR = directory that contains the dataset
* LEARNING_RATE = learning rate used in the optimizer
* OUTPUT_MODEL = the name of the model to be saved after training
* USE_CLASS_WEIGHTS = (True of False) if True uses the class weights else doesnot use the class weights

```
!python train.py --MODEL=yolo --LEARNING_RATE=0.0001 --EPOCHS=10 --BATCH_SIZE=4 --DATA_DIR=/content/drive/MyDrive/docClassification/dataset --OUTPUT_MODEL=vgg11_e20 --USE_CLASS_WEIGHTS=False
```

## Note
The classification report are saved at `./logs`

### Usage
Run tensorboard to see different metrics

```
%load_ext tensorboard
%tensorboard --logdir runs
```

## Inference
You can run inference as:
```
python inferece.py --model_path=path/to_your_model/model_name.onnx --image_path=path/to_the_document_image/
```




