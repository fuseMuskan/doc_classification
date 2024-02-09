import json
import torch
from torchvision import transforms
from PIL import Image
import onnxruntime
import numpy as np
from pathlib import Path
import argparse
import os
import shutil
from tqdm import tqdm
from torch import Tensor
from typing import List



def transform_image(image_path: str) -> Tensor:
    """
    Transform the input image to the appropriate format for a neural network model.

    Args:
        image_path (str): The path to the input image file.

    Returns:
        Tensor: The transformed image tensor ready to be fed into the neural network model.

    Raises:
        FileNotFoundError: If the specified image file is not found.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file '{image_path}' not found.")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # Apply transformations
    img = transform(img)

    # Add batch dimension to the image
    img = img.unsqueeze(0)

    # Move the image to the appropriate device (CPU or GPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.to(device)

    return img



def classify_image(model_path: str, image_path: str) -> str:
    """
    Classify an image using an ONNX model.

    Args:
        model_path (str): The path to the ONNX model file.
        image_path (str): The path to the input image file.

    Returns:
        str: The predicted class name.

    Raises:
        FileNotFoundError: If the specified model or image file is not found.
    """
    try:
        # Load model
        session = onnxruntime.InferenceSession(model_path, None)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Get input name from the model
    input_name = session.get_inputs()[0].name

    # Get output name from the model
    output_name = session.get_outputs()[0].name

    # Transform image for the model
    img = transform_image(image_path)

    # Convert image into list
    data = json.dumps({"data": img.tolist()})
    data = np.array(json.loads(data)["data"]).astype("float32")

    # Feed the image into model
    result = session.run([output_name], {input_name: data})
    max_index = np.argmax(result)
    class_names: List[str] = ["citizenship", "license", "others", "passport"]

    predicted_class = class_names[max_index]
    return predicted_class



def classify_images_in_directory(model_path: str, directory_path: str) -> None:
    """
    Classify images in a directory using an ONNX model and organize them into subdirectories based on predicted classes.

    Args:
        model_path (str): The path to the ONNX model file.
        directory_path (str): The path to the directory containing input images.

    Returns:
        None
    """
    class_names: List[str] = ["citizenship", "license", "others", "passport"]

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Filter images with supported extensions
    image_files = [filename for filename in os.listdir(directory_path)]

    # Get the total number of files in the directory
    total_files = len(image_files)

    # Initialize tqdm to track progress
    with tqdm(total=total_files, desc="Classifying Images") as pbar:
        # Iterate over images in the directory
        for filename in image_files:
            image_path = os.path.join(directory_path, filename)
            predicted_class = classify_image(model_path, image_path)
            output_class_dir = os.path.join(output_dir, predicted_class)
            os.makedirs(output_class_dir, exist_ok=True)
            output_image_path = os.path.join(output_class_dir, filename)
            shutil.copy(image_path, output_image_path)
            pbar.update(1)  # Update progress bar
            pbar.set_postfix({"Processed": filename})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path of your onnx model")
    parser.add_argument("--directory_path", type=str, help="path of the directory containing image files")

    args = parser.parse_args()

    model_path = args.model_path
    directory_path = args.directory_path

    model_path = Path(model_path)
    directory_path = Path(directory_path)
    
    print("--" * 10)
    print("Running Inference")
    print("--" * 10)
    classify_images_in_directory(model_path, directory_path)
