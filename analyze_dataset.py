"""
Comprehensive analysis of HAM10000 dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Dataset paths
DATASET_DIR = "./dataset"
METADATA_FILE = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")
IMAGES_PART1 = os.path.join(DATASET_DIR, "HAM10000_images_part_1")
IMAGES_PART2 = os.path.join(DATASET_DIR, "HAM10000_images_part_2")

# Class mapping
CLASS_MAPPING = {
    'akiec': 'Actinic Keratosis',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Nevus',
    'vasc': 'Vascular Lesion'
}

def load_metadata():
    """Load and preprocess metadata"""
    print("Loading metadata...")
    df = pd.read_csv(METADATA_FILE)
    
    # Map diagnosis codes to full names
    df['dx_name'] = df['dx'].map(CLASS_MAPPING)
    
    return df

def basic_statistics(df):
    """Print basic dataset statistics"""
    print("\n" + "="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)
    
    print(f"\nTotal number of images: {len(df)}")
    print(f"Number of unique lesions: {df['lesion_id'].nunique()}")
    print(f"Number of unique images: {df['image_id'].nunique()}")
    print(f"Number of classes: {df['dx'].nunique()}")
    
    print(f"\nMetadata columns: {list(df.columns)}")
    print(f"\nDataset shape: {df.shape}")
    
    print("\nFirst few rows:")
    print(df.head())
    
    print("\nData types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())

def class_distribution(df):
    """Analyze class distribution"""
    print("\n" + "="*60)
    print("CLASS DISTRIBUTION")
    print("="*60)
    
    class_counts = df['dx_name'].value_counts()
    class_percentages = df['dx_name'].value_counts(normalize=True) * 100
    
    distribution_df = pd.DataFrame({
        'Count': class_counts,
        'Percentage': class_percentages
    })
    
    print("\nClass distribution:")
    print(distribution_df)
    
    # Visualize
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    class_counts.plot(kind='bar', color='steelblue')
    plt.title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.subplot(1, 2, 2)
    class_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    plt.ylabel('')
    
    plt.tight_layout()
    plt.savefig('./dataset_analysis_class_distribution.png', dpi=300, bbox_inches='tight')
    print("\nClass distribution plot saved as 'dataset_analysis_class_distribution.png'")
    
    return distribution_df

def demographic_analysis(df):
    """Analyze demographic information"""
    print("\n" + "="*60)
    print("DEMOGRAPHIC ANALYSIS")
    print("="*60)
    
    # Age analysis
    print("\nAge Statistics:")
    print(f"Mean age: {df['age'].mean():.2f} years")
    print(f"Median age: {df['age'].median():.2f} years")
    print(f"Min age: {df['age'].min():.0f} years")
    print(f"Max age: {df['age'].max():.0f} years")
    print(f"Missing age values: {df['age'].isnull().sum()} ({df['age'].isnull().sum()/len(df)*100:.2f}%)")
    
    # Gender distribution
    print("\nGender Distribution:")
    gender_counts = df['sex'].value_counts()
    print(gender_counts)
    print(f"\nGender percentages:")
    print(df['sex'].value_counts(normalize=True) * 100)
    
    # Localization analysis
    print("\nTop 10 Body Localizations:")
    localization_counts = df['localization'].value_counts().head(10)
    print(localization_counts)
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Age distribution
    axes[0, 0].hist(df['age'].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Age Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Age (years)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Gender distribution
    gender_counts.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Gender Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Gender', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].tick_params(axis='x', rotation=0)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Top localizations
    localization_counts.plot(kind='barh', ax=axes[1, 0], color='lightgreen')
    axes[1, 0].set_title('Top 10 Body Localizations', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Count', fontsize=12)
    axes[1, 0].set_ylabel('Localization', fontsize=12)
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Age by class
    df_age_class = df[df['age'].notna()].groupby('dx_name')['age'].mean().sort_values()
    df_age_class.plot(kind='barh', ax=axes[1, 1], color='orange')
    axes[1, 1].set_title('Average Age by Disease Class', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Average Age (years)', fontsize=12)
    axes[1, 1].set_ylabel('Disease Class', fontsize=12)
    axes[1, 1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./dataset_analysis_demographics.png', dpi=300, bbox_inches='tight')
    print("\nDemographic analysis plot saved as 'dataset_analysis_demographics.png'")

def diagnosis_type_analysis(df):
    """Analyze diagnosis type"""
    print("\n" + "="*60)
    print("DIAGNOSIS TYPE ANALYSIS")
    print("="*60)
    
    dx_type_counts = df['dx_type'].value_counts()
    print("\nDiagnosis Type Distribution:")
    print(dx_type_counts)
    
    print("\nDiagnosis Type by Class:")
    dx_type_class = pd.crosstab(df['dx_name'], df['dx_type'])
    print(dx_type_class)
    
    # Visualization
    plt.figure(figsize=(12, 6))
    dx_type_class.plot(kind='bar', stacked=True)
    plt.title('Diagnosis Type Distribution by Class', fontsize=14, fontweight='bold')
    plt.xlabel('Disease Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Diagnosis Type')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('./dataset_analysis_diagnosis_type.png', dpi=300, bbox_inches='tight')
    print("\nDiagnosis type plot saved as 'dataset_analysis_diagnosis_type.png'")

def class_demographic_cross_analysis(df):
    """Cross-analysis of classes with demographics"""
    print("\n" + "="*60)
    print("CLASS vs DEMOGRAPHICS CROSS-ANALYSIS")
    print("="*60)
    
    # Class vs Gender
    print("\nClass Distribution by Gender:")
    class_gender = pd.crosstab(df['dx_name'], df['sex'], normalize='columns') * 100
    print(class_gender.round(2))
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Class by Gender
    class_gender_count = pd.crosstab(df['dx_name'], df['sex'])
    class_gender_count.plot(kind='bar', ax=axes[0], color=['lightblue', 'lightpink'])
    axes[0].set_title('Class Distribution by Gender', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Disease Class', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].legend(title='Gender')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Class by Top Localizations
    top_localizations = df['localization'].value_counts().head(5).index
    df_top_loc = df[df['localization'].isin(top_localizations)]
    class_loc = pd.crosstab(df_top_loc['dx_name'], df_top_loc['localization'])
    class_loc.plot(kind='bar', ax=axes[1], width=0.8)
    axes[1].set_title('Class Distribution by Top 5 Localizations', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Disease Class', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].legend(title='Localization', fontsize=8)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./dataset_analysis_cross_analysis.png', dpi=300, bbox_inches='tight')
    print("\nCross-analysis plot saved as 'dataset_analysis_cross_analysis.png'")

def data_quality_check(df):
    """Check data quality issues"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    # Check for duplicate image IDs
    duplicate_images = df[df.duplicated(subset=['image_id'], keep=False)]
    if len(duplicate_images) > 0:
        print(f"\n⚠️  Found {len(duplicate_images)} duplicate image IDs")
    else:
        print("\n✓ No duplicate image IDs found")
    
    # Check for missing values
    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Check image files existence
    print("\nChecking image files...")
    all_images = []
    if os.path.exists(IMAGES_PART1):
        part1_images = [f.replace('.jpg', '') for f in os.listdir(IMAGES_PART1) if f.endswith('.jpg')]
        all_images.extend(part1_images)
        print(f"Found {len(part1_images)} images in part_1")
    
    if os.path.exists(IMAGES_PART2):
        part2_images = [f.replace('.jpg', '') for f in os.listdir(IMAGES_PART2) if f.endswith('.jpg')]
        all_images.extend(part2_images)
        print(f"Found {len(part2_images)} images in part_2")
    
    print(f"Total images found: {len(all_images)}")
    
    # Check if metadata images exist in folders
    metadata_images = set(df['image_id'].unique())
    folder_images = set(all_images)
    
    missing_in_folders = metadata_images - folder_images
    extra_in_folders = folder_images - metadata_images
    
    if missing_in_folders:
        print(f"\n⚠️  {len(missing_in_folders)} images in metadata not found in folders")
    else:
        print("\n✓ All metadata images found in folders")
    
    if extra_in_folders:
        print(f"ℹ️  {len(extra_in_folders)} extra images in folders not in metadata")

def generate_summary_report(df):
    """Generate a summary report"""
    print("\n" + "="*60)
    print("DATASET SUMMARY REPORT")
    print("="*60)
    
    report = f"""
HAM10000 Dataset Analysis Summary
{'='*60}

DATASET OVERVIEW:
- Total Images: {len(df):,}
- Unique Lesions: {df['lesion_id'].nunique():,}
- Unique Images: {df['image_id'].nunique():,}
- Number of Classes: {df['dx'].nunique()}

CLASS DISTRIBUTION:
"""
    for dx_code, dx_name in CLASS_MAPPING.items():
        count = len(df[df['dx'] == dx_code])
        pct = (count / len(df)) * 100
        report += f"  - {dx_name:30s}: {count:5,} ({pct:5.2f}%)\n"
    
    report += f"""
DEMOGRAPHICS:
- Average Age: {df['age'].mean():.2f} years
- Age Range: {df['age'].min():.0f} - {df['age'].max():.0f} years
- Gender Distribution:
  * Male: {len(df[df['sex'] == 'male']):,} ({len(df[df['sex'] == 'male'])/len(df)*100:.2f}%)
  * Female: {len(df[df['sex'] == 'female']):,} ({len(df[df['sex'] == 'female'])/len(df)*100:.2f}%)

DATA QUALITY:
- Missing Age Values: {df['age'].isnull().sum():,} ({df['age'].isnull().sum()/len(df)*100:.2f}%)
- Missing Gender Values: {df['sex'].isnull().sum():,} ({df['sex'].isnull().sum()/len(df)*100:.2f}%)
- Missing Localization: {df['localization'].isnull().sum():,} ({df['localization'].isnull().sum()/len(df)*100:.2f}%)

TOP LOCALIZATIONS:
"""
    top_loc = df['localization'].value_counts().head(5)
    for loc, count in top_loc.items():
        report += f"  - {loc:30s}: {count:5,}\n"
    
    report += f"""
DIAGNOSIS TYPES:
"""
    for dx_type, count in df['dx_type'].value_counts().items():
        report += f"  - {dx_type:30s}: {count:5,}\n"
    
    print(report)
    
    # Save report to file
    with open('./dataset_analysis_report.txt', 'w') as f:
        f.write(report)
    print("\nFull report saved to 'dataset_analysis_report.txt'")

def main():
    """Main analysis function"""
    print("="*60)
    print("HAM10000 DATASET ANALYSIS")
    print("="*60)
    
    # Load data
    df = load_metadata()
    
    # Run analyses
    basic_statistics(df)
    class_dist = class_distribution(df)
    demographic_analysis(df)
    diagnosis_type_analysis(df)
    class_demographic_cross_analysis(df)
    data_quality_check(df)
    generate_summary_report(df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - dataset_analysis_class_distribution.png")
    print("  - dataset_analysis_demographics.png")
    print("  - dataset_analysis_diagnosis_type.png")
    print("  - dataset_analysis_cross_analysis.png")
    print("  - dataset_analysis_report.txt")

if __name__ == "__main__":
    main()


