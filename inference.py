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



def transform_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

    # Apply transformations
    img = transform(img)

    # Add batch dimension to the image
    img = img.unsqueeze(0)

    # Move the image to the appropriate device (CPU or GPU)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img = img.to(device)
    return img


def classify_image(model_path, image_path):

    # load model
    session = onnxruntime.InferenceSession(model_path, None)

    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # get input name from the model
    input_name = session.get_inputs()[0].name

    # get output name from the model
    output_name = session.get_outputs()[0].name

    # transform image for the model
    img = transform_image(image_path)

    # convert image into list
    data = json.dumps({"data": img.tolist()})
    data = np.array(json.loads(data)["data"]).astype("float32")

    # feed the image into model
    result = session.run([output_name], {input_name: data})
    max_index = np.argmax(result)
    class_names = ["citizenship", "license", "others", "passport"]

    predicted_class = class_names[max_index]
    return predicted_class


def classify_images_in_directory(model_path, directory_path):
    class_names = ["citizenship", "license", "others", "passport"]
    
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
    with tqdm(total=total_files, desc="Copying Images") as pbar:
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
