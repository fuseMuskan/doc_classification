"""
Script for running data transformation on training images
"""
import argparse
from pathlib import Path
import cv2
import os
import numpy as np
from tqdm import tqdm



# Extracting argparse values
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, help="path/to/your/data/directory")
parser.add_argument("--num_aug_images", type=int, help="number of augmented images to generate for each image")
parser.add_argument("--output_dir", type=str, help="path/to/save/directory")

args = parser.parse_args()

data_dir = args.data_dir
num_augmented_images = args.num_aug_images
output_dir = args.output_dir

# Specify the path to the dataset
dataset_dir = Path(data_dir)
output_dir = Path(output_dir)

# Expected document classes in dataset path
document_classes = ["citizenship", "license", "passport", "others"]
# Setup subdirectories path
sub_directories = [os.path.join(dataset_dir, class_name) for class_name in document_classes]
output_directories = [os.path.join(output_dir, class_name) for class_name in document_classes]


for  source_dir, output_path, class_name in zip(sub_directories, output_directories,document_classes):
    print(f"[INFO] Applying transformation to {class_name} classes")

    os.makedirs(output_path, exist_ok=True)

    # List all image files in the dataset path
    image_files = [f for f in os.listdir(
        source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


    for image_file in image_files:
        image_path = os.path.join(source_dir, image_file)

        # Read the image
        original_image = cv2.imread(image_path)

        for i in tqdm(range(num_augmented_images), desc=f"Augmenting {image_file}"):
            # Rotation
            angle = np.random.randint(-10, 10)
            rotated_image = cv2.warpAffine(original_image, cv2.getRotationMatrix2D(
                (original_image.shape[1] // 2, original_image.shape[0] // 2), angle, 1.0), (original_image.shape[1], original_image.shape[0]))

            # Scaling
            scale_factor = np.random.uniform(0.8, 1.2)
            scaled_image = cv2.resize(original_image, None, fx=scale_factor,
                                    fy=scale_factor, interpolation=cv2.INTER_LINEAR)

            # Translation
            tx, ty = np.random.randint(-20, 20, 2)
            translated_image = cv2.warpAffine(original_image, np.float32(
                [[1, 0, tx], [0, 1, ty]]), (original_image.shape[1], original_image.shape[0]))

            # Flipping
            flip_direction = np.random.randint(0, 2)
            flipped_image = cv2.flip(original_image, flip_direction)

            # Brightness adjustment
            brightness = np.random.uniform(0.5, 1.5)
            brightened_image = cv2.convertScaleAbs(
                original_image, alpha=brightness)

            # Shear
            shear_factor = np.random.uniform(-0.2, 0.2)
            shear_matrix = np.array([[1, shear_factor, 0],
                                    [0, 1, 0]], dtype=np.float32)
            sheared_image = cv2.warpAffine(
                original_image, shear_matrix, (original_image.shape[1], original_image.shape[0]))

            # Zoom
            zoom_factor = np.random.uniform(0.8, 1.2)
            zoomed_image = cv2.resize(original_image, None, fx=zoom_factor,
                                    fy=zoom_factor, interpolation=cv2.INTER_LINEAR)

            # Crop
            crop_size = np.random.uniform(0.5, 1.0)  # Adjust as needed
            crop_x = int(original_image.shape[1] * (1 - crop_size) / 2)
            crop_y = int(original_image.shape[0] * (1 - crop_size) / 2)
            cropped_image = original_image[crop_y:crop_y +
                                        int(original_image.shape[0] * crop_size),
                                        crop_x:crop_x + int(original_image.shape[1] * crop_size)]

            # Erase
            erase_ratio = np.random.uniform(0.1, 0.3)
            erase_height = int(original_image.shape[0] * erase_ratio)
            erase_width = int(original_image.shape[1] * erase_ratio)
            erase_x = np.random.randint(0, original_image.shape[1] - erase_width)
            erase_y = np.random.randint(0, original_image.shape[0] - erase_height)
            erased_image = original_image.copy()
            erased_image[erase_y:erase_y + erase_height,
                        erase_x:erase_x + erase_width, :] = 0

            # Elastic
            elastic_alpha = 50
            elastic_sigma = 10
            elastic_image = cv2.addWeighted(original_image, 4, cv2.GaussianBlur(
                original_image, (0, 0), elastic_sigma), 0.5, 0) + cv2.GaussianBlur(
                    original_image, (0, 0), elastic_sigma) * elastic_alpha

            # Save augmented images
            cv2.imwrite(os.path.join(output_path, f"rotated_{i}__{image_file}"), rotated_image)
            cv2.imwrite(os.path.join(output_path, f"scaled_{i}__{image_file}"), scaled_image)
            cv2.imwrite(os.path.join(output_path, f"translated_{i}__{image_file}"), translated_image)
            cv2.imwrite(os.path.join(output_path, f"flipped_{i}__{image_file}"), flipped_image)
            cv2.imwrite(os.path.join(output_path, f"brightened_{i}__{image_file}"), brightened_image)
            cv2.imwrite(os.path.join(output_path, f"sheared_{i}__{image_file}"), sheared_image)
            cv2.imwrite(os.path.join(output_path, f"zoomed_{i}__{image_file}"), zoomed_image)
            cv2.imwrite(os.path.join(output_path, f"cropped_{i}__{image_file}"), cropped_image)
            cv2.imwrite(os.path.join(output_path, f"erased_{i}__{image_file}"), erased_image)
            cv2.imwrite(os.path.join(output_path, f"elastic_{i}__{image_file}"), elastic_image)
    print(f"[INFO] {class_name} Augmentation Finished")
