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
    print_header("STEP 1: Exploring Dataset")
    
    if not os.path.exists("./dataset/HAM10000_metadata.csv"):
        print("[ERROR] Dataset metadata not found at ./dataset/HAM10000_metadata.csv")
        print("Please ensure the dataset is in the ./dataset/ directory")
        return False
    
    print("Running dataset analysis...")
    print("This will generate analysis reports and visualizations.\n")
    
    try:
        subprocess.run([sys.executable, "analyze_dataset.py"], check=True)
        print("\n[SUCCESS] Dataset exploration complete!")
        print("Check the generated files:")
        print("  - dataset_analysis_class_distribution.png")
        print("  - dataset_analysis_demographics.png")
        print("  - dataset_analysis_report.txt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error running dataset analysis: {e}")
        return False


def prepare_data():
    """STEP 2: Prepare and organize the dataset"""
    print_header("STEP 2: Preparing Dataset")
    
    if not os.path.exists("./dataset/HAM10000_metadata.csv"):
        print("[ERROR] Dataset metadata not found")
        return False
    
    print("Organizing images into class folders...")
    print("This may take a few minutes depending on dataset size.\n")
    
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
            print("\n[SUCCESS] Data preparation complete!")
            print(f"Processed images are in: ./data/processed/")
            return True
        else:
            print("[WARNING] No processed images found. Please check the logs above.")
            return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error preparing data: {e}")
        return False


def train_model(epochs=50):
    """STEP 3: Train the model"""
    print_header("STEP 3: Training Model")
    
    processed_dir = Path("./data/processed")
    if not processed_dir.exists() or not any(processed_dir.iterdir()):
        print("[ERROR] Processed data not found!")
        print("Please run data preparation first: python workflow.py --prepare")
        return False
    
    print(f"Starting model training with {epochs} epochs...")
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
            print("\n[SUCCESS] Model training complete!")
            print("Model saved to: ./models/skin_disease_model.h5")
            return True
        else:
            print("[WARNING] Model file not found after training")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error during training: {e}")
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

STEP 1: Explore dataset
  python workflow.py --explore

STEP 2: Prepare data
  python workflow.py --prepare

STEP 3: Train model
  python workflow.py --train
  python workflow.py --train --epochs 30  # Custom epochs

STEP 4: Test API
  python workflow.py --test

RUN ALL STEPS IN SEQUENCE:
  python workflow.py --all
        """
    )
    
    parser.add_argument("--check-deps", action="store_true", help="STEP 0: Check dependencies (FIRST STEP - Run this first!)")
    parser.add_argument("--explore", action="store_true", help="STEP 1: Explore dataset")
    parser.add_argument("--prepare", action="store_true", help="STEP 2: Prepare dataset")
    parser.add_argument("--train", action="store_true", help="STEP 3: Train model")
    parser.add_argument("--test", action="store_true", help="STEP 4: Show API testing instructions")
    parser.add_argument("--all", action="store_true", help="Run all steps in sequence (0->1->2->3->4)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    
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
    
    # STEP 3: Train model
    if args.train or args.all:
        if not train_model(epochs=args.epochs):
            print("\n" + "="*70)
            print("[ERROR] STEP 3 FAILED: Model training failed!")
            print("="*70)
            print("\n[WORKFLOW STOPPED] Cannot continue to next step.")
            return
    
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
        print("\nFinal Steps:")
        print("1. Start the API: python run.py")
        print("2. Test predictions: http://localhost:8000/docs")
        print("3. Upload images and get predictions!")


if __name__ == "__main__":
    main()


