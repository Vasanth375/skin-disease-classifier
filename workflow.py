"""
Complete workflow script for skin disease classification

STEP-BY-STEP WORKFLOW (Run in this order):
==========================================
STEP 0: Check Dependencies (FIRST - Must run this first!)
       python workflow.py --check-deps

STEP 1: Explore the Dataset
       python workflow.py --explore

STEP 2: Prepare the Data
       python workflow.py --prepare

STEP 3: Train the Model
       python workflow.py --train

STEP 4: Test the API
       python workflow.py --test

OR run all steps in sequence:
       python workflow.py --all
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_dependencies():
    """STEP 0: Check if required dependencies are installed (FIRST STEP!)"""
    print_header("STEP 0: Checking Dependencies (FIRST STEP)")
    
    # Map pip package names to their import names
    package_import_map = {
        'fastapi': 'fastapi',
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'pillow': 'PIL',  # pillow package is imported as PIL
        'opencv-python': 'cv2',  # opencv-python package is imported as cv2
        'scikit-learn': 'sklearn',  # scikit-learn package is imported as sklearn
        'matplotlib': 'matplotlib',  # Required for analyze_dataset.py
        'seaborn': 'seaborn'  # Required for analyze_dataset.py
    }
    
    missing = []
    for package_name, import_name in package_import_map.items():
        try:
            __import__(import_name)
            print(f"[OK] {package_name}")
        except ImportError:
            print(f"[MISSING] {package_name}")
            missing.append(package_name)
    
    if missing:
        print(f"\nWarning: Missing packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("\nAll dependencies are installed!")
    return True


def explore_dataset():
    """STEP 1: Run dataset exploration"""
    print_header("STEP 1: Exploring Datasets")
    
    success_count = 0
    total_steps = 0
    
    # Explore HAM10000 dataset
    if os.path.exists("./dataset/HAM10000_metadata.csv"):
        print("\n[1/3] Exploring HAM10000 dataset...")
        print("This will generate analysis reports and visualizations.\n")
        total_steps += 1
        try:
            subprocess.run([sys.executable, "analyze_dataset.py"], check=True)
            print("\n[SUCCESS] HAM10000 dataset exploration complete!")
            print("Check the generated files:")
            print("  - dataset_analysis_class_distribution.png")
            print("  - dataset_analysis_demographics.png")
            print("  - dataset_analysis_report.txt")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Error running HAM10000 analysis: {e}")
    else:
        print("\n[SKIP] HAM10000 dataset metadata not found (optional)")
    
    # Explore New Dataset 2 and 3
    print("\n[2/3] Exploring New Dataset 2 and New Dataset 3...")
    total_steps += 1
    try:
        subprocess.run([sys.executable, "analyze_new_datasets.py"], check=True)
        print("\n[SUCCESS] New datasets exploration complete!")
        print("Check the generated files:")
        print("  - new_dataset_2_analysis.png")
        print("  - new_dataset_3_analysis.png")
        print("  - new_datasets_analysis_report.txt")
        success_count += 1
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Error running new datasets analysis: {e}")
        print("This is optional - datasets may not be present yet.")
    
    if success_count == 0:
        print("\n[ERROR] No datasets found to explore!")
        print("Please ensure at least one dataset is present in ./dataset/")
        return False
    
    print(f"\n[SUCCESS] Dataset exploration complete! ({success_count}/{total_steps} steps successful)")
    return True


def prepare_data():
    """STEP 2: Prepare and organize the datasets"""
    print_header("STEP 2: Preparing Datasets")
    
    success_count = 0
    total_steps = 0
    
    # Prepare HAM10000 dataset
    if os.path.exists("./dataset/HAM10000_metadata.csv"):
        print("\n[1/3] Preparing HAM10000 dataset...")
        print("Organizing images into class folders...")
        print("This may take a few minutes depending on dataset size.\n")
        total_steps += 1
        try:
            subprocess.run([
                sys.executable, 
                "ml/training/prepare_data.py",
                "--dataset-dir", "./dataset",
                "--processed-dir", "./data/processed"
            ], check=True)
            
            # Check if data was organized
            processed_dir = Path("./data/processed")
            if processed_dir.exists() and any(processed_dir.iterdir()):
                print("\n[SUCCESS] HAM10000 data preparation complete!")
                print(f"Processed images are in: ./data/processed/")
                success_count += 1
            else:
                print("[WARNING] No processed HAM10000 images found. Please check the logs above.")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Error preparing HAM10000 data: {e}")
    else:
        print("\n[SKIP] HAM10000 dataset metadata not found (optional)")
    
    # Validate New Dataset 2 (already organized, just check)
    new_dataset_2_dir = Path("./dataset/New Dataset 2/SkinDisNet_3/Processed")
    if new_dataset_2_dir.exists() and any(new_dataset_2_dir.iterdir()):
        print("\n[2/3] Validating New Dataset 2...")
        total_steps += 1
        class_folders = [d for d in new_dataset_2_dir.iterdir() if d.is_dir()]
        total_images = 0
        for folder in class_folders:
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
            total_images += len(images)
        
        if total_images > 0:
            print(f"[SUCCESS] New Dataset 2 validated: {len(class_folders)} classes, {total_images} images")
            print(f"Dataset location: {new_dataset_2_dir}")
            success_count += 1
        else:
            print("[WARNING] New Dataset 2 folders found but no images detected")
    else:
        print("\n[SKIP] New Dataset 2 not found (optional)")
    
    # Validate New Dataset 3 (already organized, just check)
    new_dataset_3_train_dir = Path("./dataset/New Dataset 3/train")
    if new_dataset_3_train_dir.exists() and any(new_dataset_3_train_dir.iterdir()):
        print("\n[3/3] Validating New Dataset 3...")
        total_steps += 1
        class_folders = [d for d in new_dataset_3_train_dir.iterdir() if d.is_dir()]
        total_images = 0
        for folder in class_folders:
            images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
            total_images += len(images)
        
        if total_images > 0:
            print(f"[SUCCESS] New Dataset 3 validated: {len(class_folders)} classes, {total_images} training images")
            print(f"Dataset location: {new_dataset_3_train_dir}")
            success_count += 1
        else:
            print("[WARNING] New Dataset 3 folders found but no images detected")
    else:
        print("\n[SKIP] New Dataset 3 not found (optional)")
    
    if success_count == 0:
        print("\n[ERROR] No datasets found to prepare!")
        print("Please ensure at least one dataset is present in ./dataset/")
        return False
    
    print(f"\n[SUCCESS] Dataset preparation complete! ({success_count}/{total_steps} steps successful)")
    return True


def train_model(epochs=50, skip_if_exists=True):
    """STEP 3: Train the HAM10000 model (optional if already trained)"""
    print_header("STEP 3: Training HAM10000 Model (Optional)")
    
    # Check if model already exists
    if skip_if_exists and os.path.exists("./models/skin_disease_model.h5"):
        print("[SKIP] HAM10000 model already exists at ./models/skin_disease_model.h5")
        print("Skipping training. To retrain, delete the existing model first.")
        return True
    
    processed_dir = Path("./data/processed")
    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        print("[WARNING] Processed HAM10000 data not found!")
        print("HAM10000 model training skipped. You can train it later if needed.")
        print("To train HAM10000 model: python workflow.py --prepare (if not done) then --train")
        return True  # Don't fail, just skip
    
    print(f"Starting HAM10000 model training with {epochs} epochs...")
    print("This will take a significant amount of time (30 minutes to several hours).")
    print("Training progress will be displayed below.\n")
    
    try:
        # Import and run training
        from ml.training.train_model import SkinDiseaseModelTrainer
        
        trainer = SkinDiseaseModelTrainer(
            data_dir="./data/processed",
            model_save_path="./models/skin_disease_model.h5",
            epochs=epochs
        )
        
        trainer.train()
        
        if os.path.exists("./models/skin_disease_model.h5"):
            print("\n[SUCCESS] HAM10000 model training complete!")
            print("Model saved to: ./models/skin_disease_model.h5")
            return True
        else:
            print("[WARNING] Model file not found after training")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error during HAM10000 training: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_clinical_model(epochs=30, skip_if_exists=False):
    """Train the clinical model on New Dataset 3"""
    print_header("Training Clinical Model (New Dataset 3)")

    # Check if model already exists
    if skip_if_exists and os.path.exists("./models/clinical_skin_model.h5"):
        print("[SKIP] Clinical model already exists at ./models/clinical_skin_model.h5")
        print("Skipping training. To retrain, delete the existing model first.")
        return True

    dataset_dir = Path("./dataset/New Dataset 3/train")
    if not dataset_dir.exists() or not any(dataset_dir.iterdir()):
        print("[ERROR] Clinical dataset not found at ./dataset/New Dataset 3/train")
        print("Please ensure New Dataset 3 is present and structured as expected.")
        return False

    try:
        from ml.training.train_clinical_model import ClinicalSkinModelTrainer

        print(f"Starting clinical model training with {epochs} epochs...")
        print("This will train on New Dataset 3 (23 clinical disease classes).")
        print("Training progress will be displayed below.\n")
        
        trainer = ClinicalSkinModelTrainer(
            data_dir=str(dataset_dir),
            model_save_path="./models/clinical_skin_model.h5",
            epochs=epochs,
        )
        trainer.train()

        if os.path.exists("./models/clinical_skin_model.h5"):
            print("\n[SUCCESS] Clinical model training complete!")
            print("Clinical model saved to: ./models/clinical_skin_model.h5")
            return True
        print("[WARNING] Clinical model file not found after training")
        return False
    except Exception as e:  # noqa: BLE001
        print(f"[ERROR] Error during clinical model training: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_api():
    """STEP 4: Test the API with a sample image"""
    print_header("STEP 4: Testing API")
    
    if not os.path.exists("./models/skin_disease_model.h5"):
        print("[ERROR] Trained model not found!")
        print("Please train the model first: python workflow.py --train")
        return False
    
    print("To test the API, follow these steps:")
    print("\n1. Start the API server:")
    print("   python run.py")
    print("\n2. In another terminal, test with curl:")
    print("   curl -X POST \"http://localhost:8000/api/v1/predict\" \\")
    print("        -F \"file=@path/to/test_image.jpg\"")
    print("\n3. Or open in browser:")
    print("   http://localhost:8000/docs")
    print("\n4. Use the interactive API documentation to upload an image")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Complete workflow for skin disease classification - Run steps in numbered order!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
STEP-BY-STEP USAGE (Run in this order):
========================================
STEP 0: Check dependencies (FIRST - Required!)
  python workflow.py --check-deps

STEP 1: Explore all datasets (HAM10000 + New Dataset 2 + New Dataset 3)
  python workflow.py --explore

STEP 2: Prepare all datasets
  python workflow.py --prepare

STEP 3: Train models
  python workflow.py --train                    # Train HAM10000 model (skips if already exists)
  python workflow.py --train-clinical            # Train clinical model (New Dataset 3) - RECOMMENDED
  python workflow.py --train-clinical --epochs 30  # Custom epochs

STEP 4: Test API
  python workflow.py --test

QUICK START (if HAM10000 model already trained):
  python workflow.py --train-clinical            # Just train clinical model

RUN ALL STEPS IN SEQUENCE:
  python workflow.py --all                       # Includes HAM10000 (skips if exists)
  python workflow.py --all --train-clinical     # Also train clinical model
        """
    )
    
    parser.add_argument("--check-deps", action="store_true", help="STEP 0: Check dependencies (FIRST STEP - Run this first!)")
    parser.add_argument("--explore", action="store_true", help="STEP 1: Explore all datasets")
    parser.add_argument("--prepare", action="store_true", help="STEP 2: Prepare all datasets")
    parser.add_argument("--train", action="store_true", help="STEP 3: Train HAM10000 model (skips if already exists)")
    parser.add_argument("--train-clinical", action="store_true", help="Train clinical model on New Dataset 3 (RECOMMENDED)")
    parser.add_argument("--test", action="store_true", help="STEP 4: Show API testing instructions")
    parser.add_argument("--all", action="store_true", help="Run all steps in sequence (0->1->2->3->4), includes --train-clinical")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50 for HAM10000, 30 for clinical)")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n" + "="*70)
        print("IMPORTANT: Start with STEP 0 (--check-deps) to verify dependencies!")
        print("="*70)
        return
    
    # STEP 0: Check dependencies FIRST (always required)
    if args.check_deps or args.all:
        if not check_dependencies():
            print("\n" + "="*70)
            print("[ERROR] STEP 0 FAILED: Missing dependencies!")
            print("="*70)
            print("\nPlease install them first:")
            print("        pip install -r requirements.txt")
            print("\nThen run: python workflow.py --check-deps")
            print("\n[WORKFLOW STOPPED] Cannot continue without dependencies.")
            return
    
    # STEP 1: Explore dataset
    if args.explore or args.all:
        if not explore_dataset():
            print("\n" + "="*70)
            print("[ERROR] STEP 1 FAILED: Dataset exploration failed!")
            print("="*70)
            print("\n[WORKFLOW STOPPED] Cannot continue to next step.")
            return
    
    # STEP 2: Prepare data
    if args.prepare or args.all:
        if not prepare_data():
            print("\n" + "="*70)
            print("[ERROR] STEP 2 FAILED: Data preparation failed!")
            print("="*70)
            print("\n[WORKFLOW STOPPED] Cannot continue to next step.")
            return
    
    # STEP 3: Train model(s)
    if args.train or args.all:
        # HAM10000 training (skips if model already exists)
        train_model(epochs=args.epochs, skip_if_exists=True)
        # Don't fail if skipped - it's fine if model already exists
    
    # Train clinical model (New Dataset 3) - recommended
    if args.train_clinical or args.all:
        if not train_clinical_model(epochs=args.epochs, skip_if_exists=False):
            print("\n" + "="*70)
            print("[ERROR] Clinical model training failed!")
            print("="*70)
            if args.all:
                print("\n[WORKFLOW STOPPED] Cannot continue to next step.")
                return
            else:
                print("\nYou can retry with: python workflow.py --train-clinical")
    
    # STEP 4: Test API
    if args.test or args.all:
        if not test_api():
            print("\n" + "="*70)
            print("[ERROR] STEP 4 FAILED: API test failed!")
            print("="*70)
            print("\n[WORKFLOW STOPPED] Cannot continue.")
            return
    
    # All steps completed successfully
    if args.all:
        print_header("Workflow Complete!")
        print("[SUCCESS] All steps (0->1->2->3->4) completed successfully!")
        
        # Check if clinical model was trained
        clinical_model_exists = os.path.exists("./models/clinical_skin_model.h5")
        dermo_model_exists = os.path.exists("./models/skin_disease_model.h5")
        
        print("\nTrained Models:")
        if dermo_model_exists:
            print("  ✓ HAM10000 (dermoscopic) model: ./models/skin_disease_model.h5")
        if clinical_model_exists:
            print("  ✓ Clinical model: ./models/clinical_skin_model.h5")
        if not dermo_model_exists and not clinical_model_exists:
            print("  ⚠ No models found!")
        
        print("\nFinal Steps:")
        print("1. Start the API: python run.py")
        print("2. Test predictions: http://localhost:8000/docs")
        print("3. Upload images and get predictions!")
        if clinical_model_exists and dermo_model_exists:
            print("\nNote: Both models are available. The API will automatically")
            print("      use the model with higher confidence for each prediction.")


if __name__ == "__main__":
    main()


