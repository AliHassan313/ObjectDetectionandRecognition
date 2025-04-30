import os
import json
import shutil
import random
from tqdm import tqdm
from PIL import Image, ImageDraw

# Project directory structure - use forward slashes or escaped backslashes
PROJECT_DIR = "D:/Object_Detection_Project"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
VAL_DIR = os.path.join(IMAGES_DIR, "val")
TEST_DIR = os.path.join(IMAGES_DIR, "test")
VIS_DIR = os.path.join(DATASET_DIR, "visualization")

# Source directory
VAL2017_DIR = os.path.join(DATASET_DIR, "val2017")

# Classes of interest (10 classes as requested)
CLASSES_OF_INTEREST = [
    "person",      # Human
    "car",         # Vehicle
    "dog",         # Animal
    "cat",         # Animal
    "chair",       # Furniture
    "bicycle",     # Vehicle
    "laptop",      # Electronic
    "bus",         # Vehicle
    "bird",        # Animal
    "couch"        # Furniture
]

# Number of images per class
IMAGES_PER_CLASS = 300

def main():
    print("=" * 60)
    print("COCO Dataset Organizer")
    print("=" * 60)
    print(f"Target: 10 classes with {IMAGES_PER_CLASS} images each")
    print("Classes:", CLASSES_OF_INTEREST)
    print("=" * 60)
    
    # Create directory structure
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
    
    print(f"Created directory structure at: {PROJECT_DIR}")
    
    # Check if source directory exists
    if not os.path.exists(VAL2017_DIR):
        print(f"Error: Source directory not found at {VAL2017_DIR}")
        return
    
    # Check for annotations file
    instances_file = None
    
    # First check the original location
    original_instances = os.path.join(DATASET_DIR, "annotations", "instances_val2017.json")
    if os.path.exists(original_instances):
        instances_file = original_instances
    
    # Then check our annotations dir
    custom_instances = os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")
    if os.path.exists(custom_instances):
        instances_file = custom_instances
    
    if not instances_file:
        print("Error: No annotation file found. Please ensure the COCO annotations are downloaded.")
        return
    
    print(f"Using annotations file: {instances_file}")
    
    # Load annotations
    print("Loading annotations...")
    try:
        with open(instances_file, 'r') as f:
            annotations = json.load(f)
    except Exception as e:
        print(f"Error loading annotations: {e}")
        return
    
    # Find the category IDs for our classes of interest
    categories = annotations['categories']
    class_id_map = {}
    selected_categories = []
    
    for category in categories:
        if category['name'] in CLASSES_OF_INTEREST:
            class_id_map[category['name']] = category['id']
            selected_categories.append(category)
    
    if not class_id_map:
        print("Error: None of the requested classes were found in the annotations.")
        return
    
    print(f"Found {len(class_id_map)} classes of interest:")
    for class_name, cat_id in class_id_map.items():
        print(f"  - {class_name}: ID {cat_id}")
    
    # Build image to annotation mapping
    print("Building image index...")
    image_id_to_filename = {img['id']: img['file_name'] for img in annotations['images']}
    
    # Get images containing our classes
    print("Finding images with objects of interest...")
    class_to_images = {class_name: [] for class_name in CLASSES_OF_INTEREST}
    
    for ann in tqdm(annotations['annotations']):
        cat_id = ann['category_id']
        img_id = ann['image_id']
        
        # Check if this category is one we're interested in
        for class_name, class_id in class_id_map.items():
            if cat_id == class_id:
                # Add this image to the class's list if it exists and has a filename
                if img_id in image_id_to_filename:
                    class_to_images[class_name].append({
                        'image_id': img_id,
                        'file_name': image_id_to_filename[img_id],
                        'bbox': ann['bbox'],
                        'category_id': cat_id
                    })
                break
    
    # Print stats on available images
    print("\nAvailable images per class:")
    for class_name, images in class_to_images.items():
        print(f"  - {class_name}: {len(images)} annotations")
    
    # Select images for each class (up to the limit)
    selected_images = []
    class_counts = {class_name: 0 for class_name in CLASSES_OF_INTEREST}
    
    print("\nSelecting balanced dataset...")
    for class_name, images in class_to_images.items():
        # Shuffle to randomize selection
        random.shuffle(images)
        
        # Get unique image IDs (since one image can have multiple objects)
        unique_images = {}
        for img in images:
            if class_counts[class_name] >= IMAGES_PER_CLASS:
                break
                
            img_id = img['image_id']
            if img_id not in unique_images:
                unique_images[img_id] = img
                class_counts[class_name] += 1
        
        # Add selected images to our final list
        selected_images.extend(unique_images.values())
    
    print("\nSelected images per class:")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count}/{IMAGES_PER_CLASS} images")
    
    # Get unique images (since some images may contain multiple classes)
    unique_selected_images = {}
    for img in selected_images:
        unique_selected_images[img['image_id']] = img
    
    unique_image_list = list(unique_selected_images.values())
    print(f"\nSelected {len(unique_image_list)} unique images in total")
    
    # Shuffle and split into train/val/test (70/20/10)
    random.shuffle(unique_image_list)
    train_split = int(len(unique_image_list) * 0.7)
    val_split = int(len(unique_image_list) * 0.9)
    
    train_images = unique_image_list[:train_split]
    val_images = unique_image_list[train_split:val_split]
    test_images = unique_image_list[val_split:]
    
    print(f"Split into {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test images")
    
    # Copy images to their respective directories
    print("\nCopying images to train directory...")
    copy_count = 0
    for img in tqdm(train_images):
        source_file = os.path.join(VAL2017_DIR, img['file_name'])
        target_file = os.path.join(TRAIN_DIR, img['file_name'])
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            copy_count += 1
    print(f"Copied {copy_count} images to train directory")
    
    print("\nCopying images to validation directory...")
    copy_count = 0
    for img in tqdm(val_images):
        source_file = os.path.join(VAL2017_DIR, img['file_name'])
        target_file = os.path.join(VAL_DIR, img['file_name'])
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            copy_count += 1
    print(f"Copied {copy_count} images to validation directory")
    
    print("\nCopying images to test directory...")
    copy_count = 0
    for img in tqdm(test_images):
        source_file = os.path.join(VAL2017_DIR, img['file_name'])
        target_file = os.path.join(TEST_DIR, img['file_name'])
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            copy_count += 1
    print(f"Copied {copy_count} images to test directory")
    
    # Create a visualization of some random images
    print("\nCreating some example visualizations...")
    create_visualizations(unique_image_list[:5], class_id_map)
    
    print("\nDataset organization complete!")
    print(f"Dataset structure:")
    print(f"- Train images: {len(os.listdir(TRAIN_DIR)) if os.path.exists(TRAIN_DIR) else 0} in {TRAIN_DIR}")
    print(f"- Validation images: {len(os.listdir(VAL_DIR)) if os.path.exists(VAL_DIR) else 0} in {VAL_DIR}")
    print(f"- Test images: {len(os.listdir(TEST_DIR)) if os.path.exists(TEST_DIR) else 0} in {TEST_DIR}")
    print(f"- Visualizations: {len(os.listdir(VIS_DIR)) if os.path.exists(VIS_DIR) else 0} in {VIS_DIR}")

def create_visualizations(image_list, class_id_map):
    """Create visualizations of some example images with bounding boxes"""
    # Reverse the class_id_map
    id_to_class = {v: k for k, v in class_id_map.items()}
    
    for img in image_list:
        # Find the source image
        img_file = img['file_name']
        source_file = os.path.join(VAL2017_DIR, img_file)
        
        if not os.path.exists(source_file):
            print(f"Warning: Image not found for visualization: {source_file}")
            continue
        
        try:
            # Open the image
            pil_img = Image.open(source_file)
            draw = ImageDraw.Draw(pil_img)
            
            # Draw the bounding box
            bbox = img['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            # Convert to rectangle coordinates
            rect = [x, y, x + w, y + h]
            
            # Draw rectangle with category name
            class_name = id_to_class.get(img['category_id'], 'unknown')
            draw.rectangle(rect, outline="red", width=3)
            draw.text((rect[0], rect[1] - 15), class_name, fill="red")
            
            # Save the visualization
            vis_file = os.path.join(VIS_DIR, f"vis_{img_file}")
            pil_img.save(vis_file)
            print(f"Created visualization: {vis_file}")
            
        except Exception as e:
            print(f"Error creating visualization for {img_file}: {e}")

if __name__ == "__main__":
    main()