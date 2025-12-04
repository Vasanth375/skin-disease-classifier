"""
Comprehensive analysis of New Dataset 2 and New Dataset 3
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Dataset paths
NEW_DATASET_2_DIR = "../dataset/New Dataset 2/SkinDisNet_3/Processed"
NEW_DATASET_3_TRAIN_DIR = "../dataset/New Dataset 3/train"
NEW_DATASET_3_TEST_DIR = "../dataset/New Dataset 3/test"


def count_images_in_folder(folder_path):
    """Count images in a folder"""
    if not os.path.exists(folder_path):
        return 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    count = 0
    for file in os.listdir(folder_path):
        if os.path.splitext(file.lower())[1] in image_extensions:
            count += 1
    return count


def analyze_new_dataset_2():
    """Analyze New Dataset 2 (SkinDisNet_3)"""
    print("\n" + "="*70)
    print("NEW DATASET 2 ANALYSIS (SkinDisNet_3)")
    print("="*70)
    
    if not os.path.exists(NEW_DATASET_2_DIR):
        print(f"[WARNING] New Dataset 2 not found at {NEW_DATASET_2_DIR}")
        return None
    
    class_counts = {}
    total_images = 0
    
    # Get all class folders
    for item in os.listdir(NEW_DATASET_2_DIR):
        class_folder = os.path.join(NEW_DATASET_2_DIR, item)
        if os.path.isdir(class_folder):
            count = count_images_in_folder(class_folder)
            class_counts[item] = count
            total_images += count
    
    print(f"\nTotal images: {total_images}")
    print(f"Number of classes: {len(class_counts)}")
    print("\nClass distribution:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_images * 100) if total_images > 0 else 0
        print(f"  {class_name:40s}: {count:5d} images ({percentage:5.2f}%)")
    
    # Visualization
    if class_counts:
        plt.figure(figsize=(14, 6))
        
        plt.subplot(1, 2, 1)
        class_names = list(class_counts.keys())
        counts = list(class_counts.values())
        plt.barh(class_names, counts, color='steelblue')
        plt.title('New Dataset 2 - Class Distribution (Count)', fontsize=14, fontweight='bold')
        plt.xlabel('Number of Images', fontsize=12)
        plt.ylabel('Disease Class', fontsize=12)
        plt.tight_layout()
        
        plt.subplot(1, 2, 2)
        plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
        plt.title('New Dataset 2 - Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./new_dataset_2_analysis.png', dpi=300, bbox_inches='tight')
        print("\nClass distribution plot saved as 'new_dataset_2_analysis.png'")
    
    return class_counts


def analyze_new_dataset_3():
    """Analyze New Dataset 3"""
    print("\n" + "="*70)
    print("NEW DATASET 3 ANALYSIS")
    print("="*70)
    
    train_class_counts = {}
    test_class_counts = {}
    total_train = 0
    total_test = 0
    
    # Analyze train folder
    if os.path.exists(NEW_DATASET_3_TRAIN_DIR):
        print("\n[Training Set]")
        for item in os.listdir(NEW_DATASET_3_TRAIN_DIR):
            class_folder = os.path.join(NEW_DATASET_3_TRAIN_DIR, item)
            if os.path.isdir(class_folder):
                count = count_images_in_folder(class_folder)
                train_class_counts[item] = count
                total_train += count
        
        print(f"Total training images: {total_train}")
        print(f"Number of classes: {len(train_class_counts)}")
        print("\nTraining class distribution:")
        for class_name, count in sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_train * 100) if total_train > 0 else 0
            print(f"  {class_name:50s}: {count:5d} images ({percentage:5.2f}%)")
    
    # Analyze test folder
    if os.path.exists(NEW_DATASET_3_TEST_DIR):
        print("\n[Test Set]")
        for item in os.listdir(NEW_DATASET_3_TEST_DIR):
            class_folder = os.path.join(NEW_DATASET_3_TEST_DIR, item)
            if os.path.isdir(class_folder):
                count = count_images_in_folder(class_folder)
                test_class_counts[item] = count
                total_test += count
        
        print(f"Total test images: {total_test}")
        print(f"Number of classes: {len(test_class_counts)}")
        print("\nTest class distribution:")
        for class_name, count in sorted(test_class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_test * 100) if total_test > 0 else 0
            print(f"  {class_name:50s}: {count:5d} images ({percentage:5.2f}%)")
    
    # Visualization
    if train_class_counts:
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Training set - bar chart
        train_names = list(train_class_counts.keys())
        train_counts = list(train_class_counts.values())
        axes[0, 0].barh(train_names, train_counts, color='steelblue')
        axes[0, 0].set_title('New Dataset 3 - Training Set Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Number of Images', fontsize=12)
        axes[0, 0].set_ylabel('Disease Class', fontsize=10)
        
        # Training set - pie chart (top 10)
        top_10_train = sorted(train_class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_names = [x[0] for x in top_10_train]
        top_10_counts = [x[1] for x in top_10_train]
        axes[0, 1].pie(top_10_counts, labels=top_10_names, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('New Dataset 3 - Top 10 Training Classes', fontsize=14, fontweight='bold')
        
        # Test set - bar chart
        if test_class_counts:
            test_names = list(test_class_counts.keys())
            test_counts = list(test_class_counts.values())
            axes[1, 0].barh(test_names, test_counts, color='lightcoral')
            axes[1, 0].set_title('New Dataset 3 - Test Set Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Number of Images', fontsize=12)
            axes[1, 0].set_ylabel('Disease Class', fontsize=10)
        
        # Train vs Test comparison (if both exist)
        if train_class_counts and test_class_counts:
            common_classes = set(train_class_counts.keys()) & set(test_class_counts.keys())
            if common_classes:
                common_classes = sorted(common_classes)
                train_vals = [train_class_counts[c] for c in common_classes]
                test_vals = [test_class_counts.get(c, 0) for c in common_classes]
                x = range(len(common_classes))
                width = 0.35
                axes[1, 1].bar([i - width/2 for i in x], train_vals, width, label='Train', color='steelblue')
                axes[1, 1].bar([i + width/2 for i in x], test_vals, width, label='Test', color='lightcoral')
                axes[1, 1].set_xlabel('Disease Class', fontsize=10)
                axes[1, 1].set_ylabel('Number of Images', fontsize=12)
                axes[1, 1].set_title('Train vs Test Comparison', fontsize=14, fontweight='bold')
                axes[1, 1].set_xticks(x)
                axes[1, 1].set_xticklabels(common_classes, rotation=45, ha='right', fontsize=8)
                axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('./new_dataset_3_analysis.png', dpi=300, bbox_inches='tight')
        print("\nClass distribution plots saved as 'new_dataset_3_analysis.png'")
    
    return {
        'train': train_class_counts,
        'test': test_class_counts,
        'total_train': total_train,
        'total_test': total_test
    }


def generate_summary_report(dataset2_stats, dataset3_stats):
    """Generate a summary report"""
    print("\n" + "="*70)
    print("NEW DATASETS SUMMARY REPORT")
    print("="*70)
    
    report = f"""
New Datasets Analysis Summary
{'='*70}

NEW DATASET 2 (SkinDisNet_3):
"""
    if dataset2_stats:
        total_d2 = sum(dataset2_stats.values())
        report += f"- Total Images: {total_d2:,}\n"
        report += f"- Number of Classes: {len(dataset2_stats)}\n"
        report += "\nClass Distribution:\n"
        for class_name, count in sorted(dataset2_stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_d2 * 100) if total_d2 > 0 else 0
            report += f"  - {class_name:40s}: {count:5,} ({pct:5.2f}%)\n"
    else:
        report += "- Dataset not found or empty\n"
    
    report += f"""
NEW DATASET 3:
"""
    if dataset3_stats:
        report += f"- Total Training Images: {dataset3_stats['total_train']:,}\n"
        report += f"- Total Test Images: {dataset3_stats['total_test']:,}\n"
        report += f"- Total Images: {dataset3_stats['total_train'] + dataset3_stats['total_test']:,}\n"
        report += f"- Number of Classes: {len(dataset3_stats['train'])}\n"
        report += "\nTop 10 Classes (Training Set):\n"
        top_10 = sorted(dataset3_stats['train'].items(), key=lambda x: x[1], reverse=True)[:10]
        for class_name, count in top_10:
            pct = (count / dataset3_stats['total_train'] * 100) if dataset3_stats['total_train'] > 0 else 0
            report += f"  - {class_name:50s}: {count:5,} ({pct:5.2f}%)\n"
    else:
        report += "- Dataset not found or empty\n"
    
    report += f"""
COMBINED STATISTICS:
"""
    if dataset2_stats and dataset3_stats:
        total_all = sum(dataset2_stats.values()) + dataset3_stats['total_train'] + dataset3_stats['total_test']
        report += f"- Total Images Across All New Datasets: {total_all:,}\n"
        report += f"- Total Classes (unique): {len(set(dataset2_stats.keys()) | set(dataset3_stats['train'].keys()))}\n"
    
    print(report)
    
    # Save report to file
    with open('./new_datasets_analysis_report.txt', 'w') as f:
        f.write(report)
    print("\nFull report saved to 'new_datasets_analysis_report.txt'")


def main():
    """Main analysis function"""
    print("="*70)
    print("NEW DATASETS ANALYSIS")
    print("="*70)
    
    # Analyze both datasets
    dataset2_stats = analyze_new_dataset_2()
    dataset3_stats = analyze_new_dataset_3()
    
    # Generate summary
    generate_summary_report(dataset2_stats, dataset3_stats)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    if dataset2_stats:
        print("  - new_dataset_2_analysis.png")
    if dataset3_stats:
        print("  - new_dataset_3_analysis.png")
    print("  - new_datasets_analysis_report.txt")


if __name__ == "__main__":
    main()

