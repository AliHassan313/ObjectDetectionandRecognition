import os
import torch
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from pathlib import Path

# Project directory structure (from your existing code)
PROJECT_DIR = "D:/Object_Detection_Project"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
VAL_DIR = os.path.join(IMAGES_DIR, "val")
TEST_DIR = os.path.join(IMAGES_DIR, "test")
VIS_DIR = os.path.join(DATASET_DIR, "visualization")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")  # Directory to save trained models
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")  # Directory to save results

# Create directories if they don't exist
for directory in [MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)
    
# Classes of interest (as defined in your script)
CLASSES_OF_INTEREST = [
    "person", "car", "dog", "cat", "chair", 
    "bicycle", "laptop", "bus", "bird", "couch"
]

# YAML configuration file path
YAML_CONFIG_PATH = os.path.join(PROJECT_DIR, "dataset_config.yaml")

# 1. DATA PREPROCESSING AND DATASET PREPARATION FOR YOLO
def convert_coco_to_yolo_format():
    """
    Convert COCO format annotations to YOLO format
    YOLO format: class_id x_center y_center width height
    where all values are normalized to [0, 1]
    """
    print("Converting COCO annotations to YOLO format...")
    
    # Create YOLO format directories
    yolo_images_dir = os.path.join(PROJECT_DIR, "yolo_dataset", "images")
    yolo_labels_dir = os.path.join(PROJECT_DIR, "yolo_dataset", "labels")
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(yolo_images_dir, split), exist_ok=True)
        os.makedirs(os.path.join(yolo_labels_dir, split), exist_ok=True)
    
    # Process each split
    for split in ["train", "val", "test"]:
        # Read JSON annotations
        json_file = os.path.join(ANNOTATIONS_DIR, f"{split}_instances.json")
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found. Skipping {split} split.")
            continue
            
        with open(json_file, 'r') as f:
            coco_data = json.load(f)
        
        # Create category ID mapping
        # In YOLO format, class IDs start from 0
        category_id_to_class_id = {}
        for i, category in enumerate(coco_data['categories']):
            category_id_to_class_id[category['id']] = i
        
        # Create image ID to file name and size mapping
        image_info = {}
        for image in coco_data['images']:
            image_info[image['id']] = {
                'file_name': image['file_name'],
                'width': image['width'],
                'height': image['height']
            }
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        # Process each image
        print(f"Processing {split} split...")
        source_dir = os.path.join(IMAGES_DIR, split)
        
        for image_id, annotations in tqdm(annotations_by_image.items()):
            if image_id not in image_info:
                continue
                
            img_data = image_info[image_id]
            file_name = img_data['file_name']
            img_width = img_data['width']
            img_height = img_data['height']
            
            # Copy image to YOLO dataset directory
            source_path = os.path.join(source_dir, file_name)
            target_path = os.path.join(yolo_images_dir, split, file_name)
            
            if os.path.exists(source_path):
                # Create YOLO labels
                label_file_name = os.path.splitext(file_name)[0] + '.txt'
                label_path = os.path.join(yolo_labels_dir, split, label_file_name)
                
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        # Get class ID (convert to YOLO format)
                        class_id = category_id_to_class_id[ann['category_id']]
                        
                        # Get bounding box coordinates (convert to YOLO format)
                        x, y, w, h = ann['bbox']  # COCO format [x, y, width, height]
                        
                        # Convert to normalized center coordinates
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        norm_width = w / img_width
                        norm_height = h / img_height
                        
                        # Write to file
                        f.write(f"{class_id} {x_center} {y_center} {norm_width} {norm_height}\n")
                
                # Copy the image file
                if not os.path.exists(target_path):
                    try:
                        # Using PIL to open and resize the image to ensure consistency
                        img = Image.open(source_path)
                        # Optionally resize all images to a standard size
                        # img = img.resize((640, 640), Image.LANCZOS)
                        img.save(target_path)
                    except Exception as e:
                        print(f"Error processing {source_path}: {e}")
    
    # Create YAML configuration file for YOLO
    yaml_content = f"""
# COCO 10-class subset
path: {os.path.join(PROJECT_DIR, "yolo_dataset")}
train: images/train
val: images/val
test: images/test

# Classes
names:
"""
    
    # Add class names
    for i, class_name in enumerate(CLASSES_OF_INTEREST):
        yaml_content += f"  {i}: {class_name}\n"
        
    with open(YAML_CONFIG_PATH, 'w') as f:
        f.write(yaml_content)
    
    print(f"YOLO format dataset created at {os.path.join(PROJECT_DIR, 'yolo_dataset')}")
    print(f"YAML configuration file created at {YAML_CONFIG_PATH}")
    
    return YAML_CONFIG_PATH

# 2. MODEL SELECTION AND TRAINING FUNCTIONS
def train_yolov8_model(yaml_path, model_size='m', epochs=50, patience=10, image_size=640, batch_size=16):
    """
    Train a YOLOv8 model on the prepared dataset
    
    Args:
        yaml_path: Path to dataset YAML configuration file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        patience: Early stopping patience
        image_size: Input image size
        batch_size: Batch size for training
    
    Returns:
        Path to the trained model
    """
    print(f"Training YOLOv8-{model_size} model for {epochs} epochs...")
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train the model
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        patience=patience,
        imgsz=image_size,
        batch=batch_size,
        device=0 if torch.cuda.is_available() else 'cpu',
        project=MODELS_DIR,
        name=f'yolov8{model_size}_custom',
        pretrained=True,
        optimizer='Adam',  # Using Adam optimizer
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate as a ratio of lr0
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        box=7.5,  # Box loss gain
        cls=0.5,  # Classification loss gain
        hsv_h=0.015,  # Image HSV augmentation (hue)
        hsv_s=0.7,    # Image HSV augmentation (saturation)
        hsv_v=0.4,    # Image HSV augmentation (value)
        degrees=0.0,  # Image rotation (+/- deg)
        translate=0.1, # Image translation (+/- fraction)
        scale=0.5,    # Image scale (+/- gain)
        fliplr=0.5,   # Image horizontal flip probability
        mosaic=1.0,   # Image mosaic probability
        mixup=0.0     # Image mixup probability
    )
    
    # Get path to best model
    best_model_path = os.path.join(MODELS_DIR, f'yolov8{model_size}_custom', 'weights', 'best.pt')
    
    return best_model_path, results

# 3. MODEL EVALUATION
def evaluate_model(model_path, split='val', conf_threshold=0.25, iou_threshold=0.45):
    """
    Evaluate the trained model on validation or test set
    
    Args:
        model_path: Path to the trained model
        split: Dataset split to evaluate on ('val' or 'test')
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Evaluation metrics
    """
    print(f"Evaluating model on {split} set...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Run evaluation
    results = model.val(
        data=YAML_CONFIG_PATH,
        split=split,
        conf=conf_threshold,
        iou=iou_threshold,
        project=RESULTS_DIR,
        name=f'eval_{Path(model_path).stem}_{split}'
    )
    
    return results

# 4. VISUALIZATION FUNCTIONS
def visualize_predictions(model_path, num_images=10, conf_threshold=0.25):
    """
    Visualize model predictions on random test images
    
    Args:
        model_path: Path to the trained model
        num_images: Number of images to visualize
        conf_threshold: Confidence threshold for predictions
    """
    print(f"Visualizing predictions on {num_images} test images...")
    
    # Load the model
    model = YOLO(model_path)
    
    # Get list of test images
    test_images_dir = os.path.join(PROJECT_DIR, "yolo_dataset", "images", "test")
    test_images = [os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Select random images
    if len(test_images) > num_images:
        selected_images = np.random.choice(test_images, num_images, replace=False)
    else:
        selected_images = test_images
    
    # Create output directory
    vis_output_dir = os.path.join(RESULTS_DIR, "visualizations")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # Run prediction and save visualization
    for img_path in selected_images:
        # Run prediction
        results = model(img_path, conf=conf_threshold)
        
        # Get result image with annotations
        result_image = results[0].plot()
        
        # Save the visualization
        output_path = os.path.join(vis_output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, result_image)
    
    print(f"Visualizations saved to {vis_output_dir}")

# Function to plot training metrics
def plot_training_metrics(results):
    """Plot training metrics from training results"""
    # Create directory for metric plots
    plots_dir = os.path.join(RESULTS_DIR, "training_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Assuming results is a namespace or dictionary with relevant metrics
    # This will need to be adapted based on actual return structure from model.train()
    
    try:
        # Create plot of training and validation loss
        plt.figure(figsize=(12, 6))
        plt.plot(results.results_dict['train/box_loss'], label='train box loss')
        plt.plot(results.results_dict['val/box_loss'], label='val box loss')
        plt.plot(results.results_dict['train/cls_loss'], label='train cls loss')
        plt.plot(results.results_dict['val/cls_loss'], label='val cls loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'loss_plot.png'))
        
        # Create plot of mAP metrics
        plt.figure(figsize=(12, 6))
        plt.plot(results.results_dict['metrics/mAP50(B)'], label='mAP50')
        plt.plot(results.results_dict['metrics/mAP50-95(B)'], label='mAP50-95')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Mean Average Precision')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(plots_dir, 'map_plot.png'))
        
        print(f"Training metric plots saved to {plots_dir}")
    except Exception as e:
        print(f"Error plotting training metrics: {e}")
        print("Skipping metric plotting")

# Function to run inference on a single image
def run_inference(model_path, image_path, conf_threshold=0.25):
    """
    Run inference on a single image and visualize results
    
    Args:
        model_path: Path to the trained model
        image_path: Path to the image to run inference on
        conf_threshold: Confidence threshold for predictions
    """
    # Load the model
    model = YOLO(model_path)
    
    # Run prediction
    results = model(image_path, conf=conf_threshold)
    
    # Get result image with annotations
    result_image = results[0].plot()
    
    # Create output directory
    inference_dir = os.path.join(RESULTS_DIR, "inference")
    os.makedirs(inference_dir, exist_ok=True)
    
    # Save the visualization
    output_path = os.path.join(inference_dir, f"inference_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, result_image)
    
    print(f"Inference result saved to {output_path}")
    
    # Return detected objects
    boxes = results[0].boxes
    detected_objects = []
    
    for box in boxes:
        cls_id = int(box.cls.item())
        conf = box.conf.item()
        coords = box.xyxy[0].tolist()  # Get box coordinates in (x1, y1, x2, y2) format
        
        detected_objects.append({
            'class': CLASSES_OF_INTEREST[cls_id],
            'confidence': conf,
            'bbox': coords
        })
    
    return detected_objects

# Main function
def main():
    print("=" * 60)
    print("Object Detection with YOLOv8")
    print("=" * 60)
    
    # 1. Data Preprocessing - Convert COCO annotations to YOLO format
    yaml_path = convert_coco_to_yolo_format()
    
    # 2. Model Selection and Training
    # We'll use YOLOv8m for a good balance of accuracy and speed
    model_path, training_results = train_yolov8_model(
        yaml_path=yaml_path,
        model_size='m',  # Options: n (nano), s (small), m (medium), l (large), x (xlarge)
        epochs=50,
        patience=10,
        image_size=640,
        batch_size=16
    )
    
    # Plot training metrics
    plot_training_metrics(training_results)
    
    # 3. Model Evaluation on validation set
    val_results = evaluate_model(
        model_path=model_path,
        split='val',
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # 4. Model Evaluation on test set
    test_results = evaluate_model(
        model_path=model_path,
        split='test',
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # 5. Visualize predictions
    visualize_predictions(
        model_path=model_path,
        num_images=10,
        conf_threshold=0.25
    )
    
    print("\nModel training and evaluation complete!")
    print(f"Trained model saved at: {model_path}")
    print(f"Evaluation results saved at: {RESULTS_DIR}")
    
    # Print a summary of performance metrics
    print("\nPerformance Summary:")
    print(f"Validation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"Validation mAP@0.5-0.95: {val_results.box.map:.4f}")
    print(f"Test mAP@0.5: {test_results.box.map50:.4f}")
    print(f"Test mAP@0.5-0.95: {test_results.box.map:.4f}")

if __name__ == "__main__":
    main()