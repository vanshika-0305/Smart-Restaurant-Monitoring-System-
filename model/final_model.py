# -*- coding: utf-8 -*-
"""Final_Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sib6MV3XDzhONevTtOil2Ud2WN7o-Ogw
"""

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="WNU93LLN8FUCGoH7qUcd")
project = rf.workspace("varun-qlsfy").project("table-detection-gomjy")
version = project.version(10)
dataset = version.download("yolov8")

import os
target_root="/content/Table-Detection-10"

# Step 4: Count images in the merged dataset
def count_images(dataset_folder):
    image_counts = {}
    for split in ["train", "test", "valid"]:
        image_folder = os.path.join(dataset_folder, split, "images")
        if os.path.exists(image_folder):
            num_images = len([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
            image_counts[split] = num_images
        else:
            image_counts[split] = 0
    return image_counts

# Verify the final dataset
image_counts = count_images(target_root)

# Print final counts
for split, count in image_counts.items():
    print(f"📂 {split}/images: {count} images")

import os
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from collections import Counter

# Path to dataset.yaml
yaml_path = "/content/Table-Detection-10/data.yaml"  # Adjust the path if needed

# Load class names from dataset.yaml
with open(yaml_path, "r") as file:
    data = yaml.safe_load(file)
class_names = data["names"]  # Dictionary {0: "occupied", 1: "unoccupied_clean", ...}

# Function to count class occurrences in label files
def count_classes(label_dir):
    class_counts = Counter()

    # Read all .txt label files
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), "r") as f:
                for line in f:
                    class_id = int(line.split()[0])  # First value in each line is class ID
                    class_counts[class_id] += 1

    return class_counts

# Count classes in train, valid, and test sets
train_counts = count_classes("/content/Table-Detection-10/test/labels")  # Update with actual path
valid_counts = count_classes("/content/Table-Detection-10/valid/labels")
test_counts = count_classes("/content/Table-Detection-10/train/labels")

# Convert to dictionary with class names
train_counts_named = {class_names[k]: v for k, v in train_counts.items()}
valid_counts_named = {class_names[k]: v for k, v in valid_counts.items()}
test_counts_named = {class_names[k]: v for k, v in test_counts.items()}

# Plot the distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.barplot(x=list(train_counts_named.keys()), y=list(train_counts_named.values()), ax=axes[0])
axes[0].set_title("Train Set Class Distribution")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15)

sns.barplot(x=list(valid_counts_named.keys()), y=list(valid_counts_named.values()), ax=axes[1])
axes[1].set_title("Validation Set Class Distribution")
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=15)

sns.barplot(x=list(test_counts_named.keys()), y=list(test_counts_named.values()), ax=axes[2])
axes[2].set_title("Test Set Class Distribution")
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=15)

plt.tight_layout()
plt.show()

!pip install ultralytics

from ultralytics import YOLO
import torch


# Load a pre-trained YOLOv8 model
model = YOLO("yolov8m.pt")  # Start with a small model (YOLOv8n)

# Train the model
results=model.train(
    data="/content/Table-Detection-10/data.yaml",   # Path to your dataset.yaml file
    epochs=50,             # Adjust based on dataset size
    imgsz=640,             # Image size (can test 416, 640, or 1024)
    batch=8,               # Batch size (adjust based on GPU memory)
    workers=4,             # Number of data loading workers
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')          # Use GPU if available, otherwise "cpu"
)

metrics = model.val(save=True)  # Run validation

import os
from IPython.display import Image, display

# Correct training run folder
train_path = '/content/runs/detect/train22'

# Check what's inside
print(os.listdir(train_path))

# Display results.png if it exists
results_img = os.path.join(train_path, 'results.png')
if os.path.exists(results_img):
    display(Image(results_img))
else:
    print("results.png not found in:", train_path)

import matplotlib.pyplot as plt
import cv2
from IPython.display import display, Image

# Function to display images
def show_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    except Exception as e:
        print(f"Error displaying {image_path}: {e}")

# Paths to important evaluation results
conf_matrix_path = "/content/runs/detect/train22/confusion_matrix.png"
pr_curve_path = "/content/runs/detect/train22/PR_curve.png"  # corrected path
#results_path = "/content/runs/detect/train22/results.png"  # might not exist

# Display each metric visualization
print("Results Overview:")
show_image(results_path)

print("Confusion Matrix:")
show_image(conf_matrix_path)

print("Precision-Recall Curve:")
show_image(pr_curve_path)

results = model.predict("/content/restaurant.4.jpg", conf=0.5, save=True)

import cv2
import matplotlib.pyplot as plt

for r in results:
    im_array = r.plot()  # Plot the bounding boxes

    # Convert from BGR to RGB (matplotlib expects RGB)
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

    # Show the image
    plt.figure(figsize=(10, 6))
    plt.imshow(im_rgb)
    plt.axis("off")
    plt.show()

import os
import matplotlib.pyplot as plt
import cv2

# Path to your test image
image_path = "/content/Table-Detection-10/train/images/CRI_CH05_1_jpg.rf.2412d1f8cb01e33460d04289b5170b1d.jpg"

# Run prediction
results = model.predict(source=image_path, save=True)

# Get directory where prediction is saved
save_dir = results[0].save_dir  # this is a Path object
image_filename = os.path.basename(image_path)  # extract just the file name

# Full path to the saved prediction image
pred_image_path = os.path.join(save_dir, image_filename)

# Load and display
pred_image = cv2.imread(pred_image_path)
pred_image = cv2.cvtColor(pred_image, cv2.COLOR_BGR2RGB)

plt.imshow(pred_image)
plt.axis('off')
plt.title("Prediction on Test Image")
plt.show()

import os
print(os.listdir("/content/runs/detect"))

import os

base_path = "/content/runs/detect"
runs = ['train', 'train22', 'train23', 'train24', 'train25', 'train26', 'train2']

for run in runs:
    weights_path = os.path.join(base_path, run, "weights")
    if os.path.exists(weights_path):
        print(f"{run} contains: {os.listdir(weights_path)}")
    else:
        print(f"{run} has no weights folder.")

import shutil
import os
from google.colab import files

# Path to the actual model
source_path = "/content/runs/detect/train2/weights/best.pt"
destination_folder = "/content/exported_model"
destination_path = os.path.join(destination_folder, "best.pt")

# Create destination folder and copy the model
os.makedirs(destination_folder, exist_ok=True)
shutil.copy(source_path, destination_path)

print(f"Model exported successfully to {destination_path}")

# Download the file
files.download(destination_path)

#PREDICTION ON TEST DATA


import glob

# Get a test image from the test dataset
test_images = glob.glob("/content/Table-Detection-10/test/images")  # Change extension if needed

if test_images:
    test_image_path = test_images[0]  # Select the first test image
    print(f"Using test image: {test_image_path}")
else:
    print("No test images found. Check the path.")

results = model.predict(test_image_path, conf=0.25, save=True)  # Change confidence if needed


import cv2
import matplotlib.pyplot as plt

predicted_image_path = "/content/Table-Detection-10/test/images"  # Correct path to your predicted image
# Check if the file exists before reading it
if os.path.exists(predicted_image_path):
    image = cv2.imread(predicted_image_path)
    if image is not None:  # Check if image was loaded successfully
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    else:
        print(f"Error: Could not read image from {predicted_image_path}")
else:
    print(f"Error: Image file not found at {predicted_image_path}")



# Get the latest predicted image
predicted_images = glob.glob("/content/runs/detect/predict/*.jpg")

if predicted_images:
    predicted_image_path = predicted_images[0]  # Pick the first predicted image
    print(f"Predicted image found: {predicted_image_path}")

    # Load and display the image
    image = cv2.imread(predicted_image_path)

    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
    else:
        print("Error: Image found but cannot be read.")
else:
    print("Error: No predicted images found. Check if YOLO saved the results.")

import shutil
import os

# Define paths
source_path = "/content/runs/detect/train/weights/best.pt"
destination_folder = "/content/exported_model"
destination_path = os.path.join(destination_folder, "best.pt")

# Create the destination directory if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Copy best.pt to the export folder
shutil.copy(source_path, destination_path)

print(f"Model exported successfully to {destination_path}")

from google.colab import files

files.download("/content/runs/detect/train/weights/best.pt")

# prompt: i want to download the best.pt model which will be stored in the detect folder

# Download the best.pt model
model.export(format="onnx")
!cp /content/runs/detect/train/weights/best.pt /content/detect/

