"""
Script to prepare and organize HAM10000 dataset
"""
import os
import pandas as pd
import shutil
from pathlib import Path
import zipfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# HAM10000 class mapping
CLASS_MAPPING = {
    'akiec': 'actinic_keratosis',
    'bcc': 'basal_cell_carcinoma',
    'bkl': 'benign_keratosis',
    'df': 'dermatofibroma',
    'mel': 'melanoma',
    'nv': 'nevus',
    'vasc': 'vascular_lesion'
}


def extract_dataset(zip_path: str, extract_to: str = "./data/raw"):
    """
    Extract HAM10000 dataset zip file
    
    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to
    """
    logger.info(f"Extracting dataset from {zip_path}...")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    logger.info(f"Dataset extracted to {extract_to}")


def organize_dataset(
    dataset_dir: str = "./dataset",
    processed_data_dir: str = "./data/processed",
    metadata_file: str = None
):
    """
    Organize dataset into class folders
    
    Args:
        dataset_dir: Directory containing dataset (with images_part_1 and images_part_2)
        processed_data_dir: Directory to organize images into
        metadata_file: Path to HAM10000 metadata CSV file
    """
    logger.info("Organizing dataset...")
    
    # Default paths
    if metadata_file is None:
        metadata_file = os.path.join(dataset_dir, "HAM10000_metadata.csv")
    
    images_part1 = os.path.join(dataset_dir, "HAM10000_images_part_1")
    images_part2 = os.path.join(dataset_dir, "HAM10000_images_part_2")
    
    # Create class directories
    for class_name in CLASS_MAPPING.values():
        class_dir = os.path.join(processed_data_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        logger.info(f"Created directory: {class_dir}")
    
    # Load metadata if available
    if metadata_file and os.path.exists(metadata_file):
        df = pd.read_csv(metadata_file)
        logger.info(f"Loaded metadata with {len(df)} entries")
        
        # Count images organized per class
        class_counts = {class_name: 0 for class_name in CLASS_MAPPING.values()}
        images_not_found = 0
        
        # Organize images based on metadata
        for idx, row in df.iterrows():
            image_id = row['image_id']
            dx = row['dx']  # Diagnosis code
            
            # Map diagnosis to class name
            class_name = CLASS_MAPPING.get(dx, 'unknown')
            
            if class_name != 'unknown':
                # Find image file in both part directories
                image_extensions = ['.jpg', '.jpeg', '.png']
                source_path = None
                
                # Check in part 1
                for ext in image_extensions:
                    possible_path = os.path.join(images_part1, f"{image_id}{ext}")
                    if os.path.exists(possible_path):
                        source_path = possible_path
                        break
                
                # Check in part 2 if not found in part 1
                if source_path is None:
                    for ext in image_extensions:
                        possible_path = os.path.join(images_part2, f"{image_id}{ext}")
                        if os.path.exists(possible_path):
                            source_path = possible_path
                            break
                
                if source_path:
                    dest_dir = os.path.join(processed_data_dir, class_name)
                    dest_path = os.path.join(dest_dir, f"{image_id}{os.path.splitext(source_path)[1]}")
                    
                    # Copy file if it doesn't already exist
                    if not os.path.exists(dest_path):
                        shutil.copy2(source_path, dest_path)
                        class_counts[class_name] += 1
                    
                    if (idx + 1) % 1000 == 0:
                        logger.info(f"Processed {idx + 1}/{len(df)} images...")
                else:
                    images_not_found += 1
        
        logger.info("\n" + "="*60)
        logger.info("Dataset organization complete!")
        logger.info("="*60)
        logger.info(f"Total images organized: {sum(class_counts.values())}")
        logger.info(f"Images not found: {images_not_found}")
        logger.info("\nClass distribution:")
        for class_name, count in class_counts.items():
            logger.info(f"  {class_name:30s}: {count:5d} images")
        logger.info("="*60)
        
    else:
        logger.warning(f"Metadata file not found at {metadata_file}")
        logger.warning("Please organize images manually or provide metadata file.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare HAM10000 dataset for training")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./dataset",
        help="Directory containing dataset (default: ./dataset)"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="./data/processed",
        help="Directory to save processed images (default: ./data/processed)"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata CSV file (default: ./dataset/HAM10000_metadata.csv)"
    )
    
    args = parser.parse_args()
    
    # Organize the dataset
    organize_dataset(
        dataset_dir=args.dataset_dir,
        processed_data_dir=args.processed_dir,
        metadata_file=args.metadata
    )

