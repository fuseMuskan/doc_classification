import json
import torch
from torchvision import transforms
from PIL import Image
import onnxruntime
import numpy as np
from pathlib import Path
import argparse


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="path of your onnx model")
    parser.add_argument("--image_path", type=str, help="path of the document image")

    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    model_path = Path(model_path)
    image_path = Path(image_path)
    print("--" * 10)
    print("Running Inference")
    print("--" * 10)
    prediction = classify_image(model_path, image_path)
    print(f"Predicted Class = {prediction}")