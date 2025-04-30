import os
import json
import shutil
import random
import urllib.request
import zipfile
from tqdm import tqdm
from PIL import Image, ImageDraw

# Project directory structure
PROJECT_DIR = "D:/Object_Detection_Project"
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")
TRAIN_DIR = os.path.join(IMAGES_DIR, "train")
VAL_DIR = os.path.join(IMAGES_DIR, "val")
TEST_DIR = os.path.join(IMAGES_DIR, "test")
VIS_DIR = os.path.join(DATASET_DIR, "visualization")

# Download URLs
VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

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

# Create all required directories
def create_directories():
    """Create all necessary directories for the project"""
    dirs = [
        DATASET_DIR, IMAGES_DIR, ANNOTATIONS_DIR,
        TRAIN_DIR, VAL_DIR, TEST_DIR, VIS_DIR
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created directory structure at: {PROJECT_DIR}")

# Helper function to download files with progress bar
def download_file(url, destination):
    """Download a file with progress reporting"""
    if os.path.exists(destination):
        print(f"File already exists: {destination}")
        return
    
    print(f"Downloading {url} to {destination}")
    
    # Create a simple progress bar
    def report_progress(count, block_size, total_size):
        progress = count * block_size * 100 / total_size
        if progress > 100:
            progress = 100
        print(f"\rProgress: {progress:.2f}%", end="")
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading file: {e}")

# Extract zip files
def extract_zip(zip_file, extract_to):
    """Extract zip file to specified directory"""
    print(f"Extracting {zip_file} to {extract_to}")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction complete!")

# Download and extract COCO dataset
def download_and_extract_dataset():
    """Download and extract COCO dataset files"""
    # Define zip file paths
    val_images_zip = os.path.join(DATASET_DIR, "val2017.zip")
    annotations_zip = os.path.join(DATASET_DIR, "annotations_trainval2017.zip")
    
    # Download files if they don't exist
    print("Downloading COCO validation images (this may take some time)...")
    download_file(VAL_IMAGES_URL, val_images_zip)
    
    print("Downloading COCO annotations...")
    download_file(ANNOTATIONS_URL, annotations_zip)
    
    # Extract files
    extract_zip(val_images_zip, DATASET_DIR)
    extract_zip(annotations_zip, DATASET_DIR)
    
    # Move annotation files to our annotations directory
    source_annotations_dir = os.path.join(DATASET_DIR, "annotations")
    if os.path.exists(source_annotations_dir):
        instances_file = os.path.join(source_annotations_dir, "instances_val2017.json")
        if os.path.exists(instances_file):
            shutil.copy(instances_file, os.path.join(ANNOTATIONS_DIR, "instances_val2017.json"))
            print(f"Copied annotations file to {ANNOTATIONS_DIR}")
    
    return os.path.join(DATASET_DIR, "val2017")

# Filter dataset by classes of interest and organize images
def filter_and_organize_dataset(source_images_dir):
    """Filter dataset by classes of interest and organize into train/val/test splits"""
    print("Filtering and organizing COCO dataset...")
    
    # Load annotations
    instances_file = os.path.join(ANNOTATIONS_DIR, "instances_val2017.json")
    if not os.path.exists(instances_file):
        print(f"Error: Annotation file not found at {instances_file}")
        return
    
    print(f"Loading annotations from {instances_file}...")
    with open(instances_file, 'r') as f:
        annotations = json.load(f)
    
    # Get category IDs for our classes of interest
    categories = annotations['categories']
    selected_category_ids = []
    selected_categories = []
    category_name_to_id = {}
    
    print("Filtering for these classes:", CLASSES_OF_INTEREST)
    for category in categories:
        if category['name'] in CLASSES_OF_INTEREST:
            category_id = category['id']
            selected_category_ids.append(category_id)
            selected_categories.append(category)
            category_name_to_id[category['name']] = category_id
    
    print(f"Found {len(selected_category_ids)} matching categories")
    
    # Find images that have our classes of interest
    # We'll keep track of how many images we have per class
    class_image_counts = {name: 0 for name in CLASSES_OF_INTEREST}
    selected_image_ids = set()
    
    # Group annotations by image_id
    image_annotations = {}
    for annotation in annotations['annotations']:
        img_id = annotation['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(annotation)
    
    # Select images with our classes of interest, ensuring we get enough per class
    print("Selecting images with balanced class distribution...")
    
    # First, count how many images we have for each class
    for img_id, anns in image_annotations.items():
        for ann in anns:
            if ann['category_id'] in selected_category_ids:
                cat_name = next((name for name, cat_id in category_name_to_id.items() 
                                 if cat_id == ann['category_id']), None)
                if cat_name and class_image_counts[cat_name] < IMAGES_PER_CLASS:
                    selected_image_ids.add(img_id)
                    class_image_counts[cat_name] += 1
    
    # Check how many images we have per class
    print("\nImages found per class:")
    for class_name, count in class_image_counts.items():
        print(f"- {class_name}: {count}/{IMAGES_PER_CLASS} images")
    
    print(f"Selected {len(selected_image_ids)} unique images")
    
    # Get the selected image details
    image_id_to_file = {}
    selected_images = []
    
    for image in annotations['images']:
        if image['id'] in selected_image_ids:
            selected_images.append(image)
            image_id_to_file[image['id']] = image['file_name']
    
    # Get the annotations for the selected images
    selected_annotations = []
    for annotation in annotations['annotations']:
        if (annotation['image_id'] in selected_image_ids and 
            annotation['category_id'] in selected_category_ids):
            selected_annotations.append(annotation)
    
    print(f"Selected {len(selected_images)} images with {len(selected_annotations)} annotations")
    
    # Create filtered annotations file
    filtered_annotations = {
        'info': annotations['info'],
        'licenses': annotations['licenses'],
        'images': selected_images,
        'annotations': selected_annotations,
        'categories': selected_categories
    }
    
    filtered_file = os.path.join(ANNOTATIONS_DIR, "filtered_instances.json")
    with open(filtered_file, 'w') as f:
        json.dump(filtered_annotations, f)
    
    print(f"Saved filtered annotations to {filtered_file}")
    
    # Split into train/val/test (70/20/10 split)
    random.shuffle(selected_images)
    train_split = int(len(selected_images) * 0.7)
    val_split = int(len(selected_images) * 0.9)
    
    train_images = selected_images[:train_split]
    val_images = selected_images[train_split:val_split]
    test_images = selected_images[val_split:]
    
    print(f"Split into {len(train_images)} training, {len(val_images)} validation, and {len(test_images)} test images")
    
    # Copy images to their respective directories
    copy_images(train_images, source_images_dir, TRAIN_DIR)
    copy_images(val_images, source_images_dir, VAL_DIR)
    copy_images(test_images, source_images_dir, TEST_DIR)
    
    # Create split-specific annotation files
    create_split_annotations(filtered_annotations, train_images, "train")
    create_split_annotations(filtered_annotations, val_images, "val")
    create_split_annotations(filtered_annotations, test_images, "test")
    
    # Visualize some examples
    visualize_examples(filtered_annotations, 5)
    
    return filtered_annotations

# Copy images from source to target directory
def copy_images(image_list, source_dir, target_dir):
    """Copy images from source to target directory"""
    print(f"Copying {len(image_list)} images to {os.path.basename(target_dir)}...")
    for image in tqdm(image_list):
        file_name = image['file_name']
        source_path = os.path.join(source_dir, file_name)
        target_path = os.path.join(target_dir, file_name)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, target_path)
        else:
            print(f"Warning: Source image not found: {source_path}")

# Create annotation files for each split
def create_split_annotations(annotations, image_subset, split_name):
    """Create annotation files specific to train/val/test splits"""
    image_ids = [img['id'] for img in image_subset]
    
    # Filter annotations for this split
    split_annotations = [a for a in annotations['annotations'] if a['image_id'] in image_ids]
    
    # Create the split-specific annotation file
    split_data = {
        'info': annotations['info'],
        'licenses': annotations['licenses'],
        'images': image_subset,
        'annotations': split_annotations,
        'categories': annotations['categories']
    }
    
    # Save to file
    split_file = os.path.join(ANNOTATIONS_DIR, f"{split_name}_instances.json")
    with open(split_file, 'w') as f:
        json.dump(split_data, f)
    
    print(f"Created {split_name} annotation file with {len(image_subset)} images and {len(split_annotations)} annotations")

# Visualize examples with bounding boxes
def visualize_examples(annotations, num_examples=5):
    """Create visualization of random examples with bounding boxes"""
    print(f"Creating {num_examples} example visualizations...")
    
    # Get categories mapping
    categories = {cat['id']: cat['name'] for cat in annotations['categories']}
    
    # Group annotations by image
    image_annotations = {}
    for ann in annotations['annotations']:
        img_id = ann['image_id']
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)
    
    # Get image id to filename mapping
    image_id_to_file = {img['id']: img['file_name'] for img in annotations['images']}
    
    # Select random images with annotations
    image_ids = list(image_annotations.keys())
    if len(image_ids) > num_examples:
        sample_ids = random.sample(image_ids, num_examples)
    else:
        sample_ids = image_ids
    
    for img_id in sample_ids:
        file_name = image_id_to_file[img_id]
        
        # Try to find the image in train, val, or test directories
        for dir_name in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            img_path = os.path.join(dir_name, file_name)
            if os.path.exists(img_path):
                break
        else:
            print(f"Image not found: {file_name}")
            continue
        
        # Open image and draw bounding boxes
        try:
            img = Image.open(img_path)
            draw = ImageDraw.Draw(img)
            
            for ann in image_annotations[img_id]:
                bbox = ann['bbox']  # [x, y, width, height]
                category_id = ann['category_id']
                category_name = categories.get(category_id, "Unknown")
                
                # Convert COCO format [x, y, width, height] to [x1, y1, x2, y2]
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                
                # Draw category name
                draw.text((x1, y1 - 15), category_name, fill="red")
            
            # Save the visualization
            vis_path = os.path.join(VIS_DIR, f"vis_{file_name}")
            img.save(vis_path)
            print(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            print(f"Error visualizing {file_name}: {e}")

# Analyze dataset statistics
def analyze_dataset():
    """Analyze the organized dataset"""
    print("\nAnalyzing organized dataset:")
    
    # Count images in each directory
    train_images = os.listdir(TRAIN_DIR)
    val_images = os.listdir(VAL_DIR)
    test_images = os.listdir(TEST_DIR)
    
    print(f"Train images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Total: {len(train_images) + len(val_images) + len(test_images)}")
    
    # Analyze class distribution in filtered annotations
    filtered_file = os.path.join(ANNOTATIONS_DIR, "filtered_instances.json")
    if os.path.exists(filtered_file):
        with open(filtered_file, 'r') as f:
            data = json.load(f)
        
        # Count annotations per category
        categories = {cat['id']: cat['name'] for cat in data['categories']}
        cat_counts = {}
        
        for ann in data['annotations']:
            cat_id = ann['category_id']
            cat_name = categories.get(cat_id, f"Unknown ({cat_id})")
            
            if cat_name not in cat_counts:
                cat_counts[cat_name] = 0
            cat_counts[cat_name] += 1
        
        print("\nClass distribution:")
        for cat_name, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {cat_name}: {count} annotations")

# Optional cleanup function
def cleanup(keep_originals=False):
    """Clean up temporary files"""
    if not keep_originals:
        print("Cleaning up temporary files...")
        
        # Files to potentially remove
        val_images_zip = os.path.join(DATASET_DIR, "val2017.zip")
        annotations_zip = os.path.join(DATASET_DIR, "annotations_trainval2017.zip")
        val2017_dir = os.path.join(DATASET_DIR, "val2017")
        original_annotations_dir = os.path.join(DATASET_DIR, "annotations")
        
        # Remove downloaded zip files
        if os.path.exists(val_images_zip):
            os.remove(val_images_zip)
            print(f"Removed {val_images_zip}")
            
        if os.path.exists(annotations_zip):
            os.remove(annotations_zip)
            print(f"Removed {annotations_zip}")
        
        # Remove extracted original directories
        if os.path.exists(val2017_dir):
            shutil.rmtree(val2017_dir)
            print(f"Removed {val2017_dir}")
            
        # Only remove original annotations dir if it's different from our annotations dir
        if os.path.exists(original_annotations_dir) and original_annotations_dir != ANNOTATIONS_DIR:
            shutil.rmtree(original_annotations_dir)
            print(f"Removed {original_annotations_dir}")

# Main function
def main():
    print("=" * 60)
    print("COCO Dataset Downloader and Organizer")
    print("=" * 60)
    print(f"Target: 10 classes with {IMAGES_PER_CLASS} images each")
    print("Classes:", CLASSES_OF_INTEREST)
    print("=" * 60)
    
    # Create directory structure
    create_directories()
    
    # Download and extract dataset
    source_images_dir = download_and_extract_dataset()
    
    # Filter and organize dataset
    filter_and_organize_dataset(source_images_dir)
    
    # Analyze dataset
    analyze_dataset()
    
    # Clean up (optional)
    cleanup(keep_originals=False)
    
    print("\nDataset preparation complete!")
    print(f"Dataset structure:")
    print(f"- Train images: {len(os.listdir(TRAIN_DIR))} in {TRAIN_DIR}")
    print(f"- Validation images: {len(os.listdir(VAL_DIR))} in {VAL_DIR}")
    print(f"- Test images: {len(os.listdir(TEST_DIR))} in {TEST_DIR}")
    print(f"- Annotations: {ANNOTATIONS_DIR}")
    print(f"- Visualizations: {VIS_DIR}")

if __name__ == "__main__":
    main()