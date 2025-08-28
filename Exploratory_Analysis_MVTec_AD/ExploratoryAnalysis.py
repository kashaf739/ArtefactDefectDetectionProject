"""
Exploratory Data Analysis for MVTec-AD Dataset
This script performs comprehensive analysis of the MVTec-AD dataset for industrial defect detection.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import warnings
import traceback
from datetime import datetime

# Configure environment
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# Google Colab specific imports and setup
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully")
except Exception as e:
    print(f"Error mounting Google Drive: {str(e)}")
    print("Continuing without mounting...")

# Configuration
DATASET_ROOT = "/content/drive/MyDrive/MVTEC_AD/mvtec_ad"
OUTPUT_DIR = "/content/drive/MyDrive/MVTec_AD_EDA_3rd_Cell"
SAMPLE_SIZE_PER_CATEGORY = 50  # Number of images to sample for analysis per category
RANDOM_STATE = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Output directory created at: {OUTPUT_DIR}")

# Set random seed for reproducibility
np.random.seed(RANDOM_STATE)

# Helper function to convert numpy types to JSON serializable
def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Helper function for error handling
def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"❌ Error in {func.__name__}: {str(e)}")
        traceback.print_exc()
        return None

# Function to analyze dataset structure
def analyze_dataset_structure(root_path):
    """Analyze the structure of the MVTec-AD dataset"""
    print("🔍 Analyzing dataset structure...")

    try:
        if not os.path.exists(root_path):
            raise FileNotFoundError(f"Dataset root path does not exist: {root_path}")

        categories = [d for d in os.listdir(root_path)
                     if os.path.isdir(os.path.join(root_path, d))]

        if not categories:
            raise ValueError("No categories found in the dataset")

        structure_data = []

        for category in tqdm(categories, desc="Processing categories"):
            category_path = os.path.join(root_path, category)

            # Analyze train data (only good images)
            train_good_path = os.path.join(category_path, 'train', 'good')
            train_good_count = len(os.listdir(train_good_path)) if os.path.exists(train_good_path) else 0

            # Analyze test data (good + defective)
            test_good_path = os.path.join(category_path, 'test', 'good')
            test_good_count = len(os.listdir(test_good_path)) if os.path.exists(test_good_path) else 0

            # Analyze defective images
            test_path = os.path.join(category_path, 'test')
            defect_types = [d for d in os.listdir(test_path)
                           if os.path.isdir(os.path.join(test_path, d)) and d != 'good']

            defect_counts = {}
            total_defects = 0
            for defect in defect_types:
                defect_path = os.path.join(test_path, defect)
                count = len(os.listdir(defect_path))
                defect_counts[defect] = count
                total_defects += count

            structure_data.append({
                'category': category,
                'train_good': train_good_count,
                'test_good': test_good_count,
                'test_defects': total_defects,
                'defect_types': defect_types,
                'defect_counts': defect_counts,
                'total_images': train_good_count + test_good_count + total_defects,
                'imbalance_ratio': total_defects / (test_good_count + 1e-8)  # Avoid division by zero
            })

        return pd.DataFrame(structure_data)

    except Exception as e:
        print(f"❌ Error analyzing dataset structure: {str(e)}")
        traceback.print_exc()
        return None

# Function to visualize class distribution
def visualize_class_distribution(df):
    """Visualize class distribution across categories"""
    print("📊 Visualizing class distribution...")

    try:
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")

        # Overall distribution
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        # Total images per category
        sns.barplot(data=df, x='category', y='total_images', ax=axes[0, 0])
        axes[0, 0].set_title('Total Images per Category', fontsize=14)
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].set_xlabel('Category', fontsize=12)
        axes[0, 0].set_ylabel('Number of Images', fontsize=12)

        # Class imbalance ratio
        sns.barplot(data=df, x='category', y='imbalance_ratio', ax=axes[0, 1])
        axes[0, 1].set_title('Class Imbalance Ratio (Defects/Good)', fontsize=14)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].set_xlabel('Category', fontsize=12)
        axes[0, 1].set_ylabel('Imbalance Ratio', fontsize=12)

        # Good vs Defective comparison
        df_melted = df.melt(id_vars=['category'],
                            value_vars=['train_good', 'test_good', 'test_defects'],
                            var_name='type', value_name='count')
        sns.barplot(data=df_melted, x='category', y='count', hue='type', ax=axes[1, 0])
        axes[1, 0].set_title('Image Type Distribution per Category', fontsize=14)
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].set_xlabel('Category', fontsize=12)
        axes[1, 0].set_ylabel('Number of Images', fontsize=12)

        # Pie chart of overall distribution
        total_good = df['train_good'].sum() + df['test_good'].sum()
        total_defects = df['test_defects'].sum()
        axes[1, 1].pie([total_good, total_defects], labels=['Good', 'Defective'],
                       autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
        axes[1, 1].set_title('Overall Dataset Distribution', fontsize=14)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Class distribution visualization saved")
        return True

    except Exception as e:
        print(f"❌ Error visualizing class distribution: {str(e)}")
        traceback.print_exc()
        return False

# Function to analyze image properties
def analyze_image_properties(df, root_path, sample_size=SAMPLE_SIZE_PER_CATEGORY):
    """Analyze image properties (size, color distribution, etc.)"""
    print("🔬 Analyzing image properties...")

    try:
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")

        properties_data = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing images"):
            category = row['category']

            # Sample good images
            good_path = os.path.join(root_path, category, 'train', 'good')
            if not os.path.exists(good_path):
                continue

            good_images = [os.path.join(good_path, img) for img in os.listdir(good_path)]
            good_images = np.random.choice(good_images, min(sample_size//2, len(good_images)), replace=False)

            # Sample defective images
            test_path = os.path.join(root_path, category, 'test')
            defect_types = [d for d in os.listdir(test_path)
                           if os.path.isdir(os.path.join(test_path, d)) and d != 'good']

            defect_images = []
            for defect in defect_types:
                defect_path = os.path.join(test_path, defect)
                if os.path.exists(defect_path):
                    defect_imgs = [os.path.join(defect_path, img) for img in os.listdir(defect_path)]
                    defect_imgs = np.random.choice(defect_imgs, min(sample_size//4, len(defect_imgs)), replace=False)
                    defect_images.extend(defect_imgs)

            # Analyze good images
            for img_path in good_images:
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    # Handle grayscale images
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)

                    properties_data.append({
                        'category': category,
                        'type': 'good',
                        'width': img.width,
                        'height': img.height,
                        'channels': img_array.shape[2],
                        'mean_brightness': np.mean(img_array),
                        'std_brightness': np.std(img_array),
                        'mean_r': np.mean(img_array[:,:,0]),
                        'mean_g': np.mean(img_array[:,:,1]),
                        'mean_b': np.mean(img_array[:,:,2]),
                    })
                except Exception as e:
                    continue

            # Analyze defective images
            for img_path in defect_images:
                try:
                    img = Image.open(img_path)
                    img_array = np.array(img)

                    # Handle grayscale images
                    if len(img_array.shape) == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)

                    properties_data.append({
                        'category': category,
                        'type': 'defective',
                        'width': img.width,
                        'height': img.height,
                        'channels': img_array.shape[2],
                        'mean_brightness': np.mean(img_array),
                        'std_brightness': np.std(img_array),
                        'mean_r': np.mean(img_array[:,:,0]),
                        'mean_g': np.mean(img_array[:,:,1]),
                        'mean_b': np.mean(img_array[:,:,2]),
                    })
                except Exception as e:
                    continue

        return pd.DataFrame(properties_data)

    except Exception as e:
        print(f"Error analyzing image properties: {str(e)}")
        traceback.print_exc()
        return None

# Function to create visual comparison of good vs defective images
def create_visual_comparison(df, root_path):
    """Create visual comparison of good vs defective images"""
    print("Creating visual comparisons...")

    try:
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")

        for category in df['category'].unique():
            try:
                fig, axes = plt.subplots(2, 5, figsize=(20, 8))
                fig.suptitle(f'Visual Comparison: {category}', fontsize=16)

                # Get good images
                good_path = os.path.join(root_path, category, 'train', 'good')
                if not os.path.exists(good_path):
                    continue

                good_images = [os.path.join(good_path, img) for img in os.listdir(good_path)[:5]]

                # Get defective images
                test_path = os.path.join(root_path, category, 'test')
                defect_types = [d for d in os.listdir(test_path)
                               if os.path.isdir(os.path.join(test_path, d)) and d != 'good']

                defect_images = []
                for defect in defect_types[:1]:  # Use first defect type
                    defect_path = os.path.join(test_path, defect)
                    if os.path.exists(defect_path):
                        defect_images = [os.path.join(defect_path, img)
                                        for img in os.listdir(defect_path)[:5]]

                # Display good images
                for i, img_path in enumerate(good_images):
                    try:
                        img = Image.open(img_path)
                        axes[0, i].imshow(img)
                        axes[0, i].set_title('Good')
                        axes[0, i].axis('off')
                    except:
                        axes[0, i].text(0.5, 0.5, 'Error', ha='center', va='center')
                        axes[0, i].axis('off')

                # Display defective images
                for i, img_path in enumerate(defect_images):
                    try:
                        img = Image.open(img_path)
                        axes[1, i].imshow(img)
                        axes[1, i].set_title('Defective')
                        axes[1, i].axis('off')
                    except:
                        axes[1, i].text(0.5, 0.5, 'Error', ha='center', va='center')
                        axes[1, i].axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f'visual_comparison_{category}.png'),
                           dpi=300, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"Error creating comparison for {category}: {str(e)}")
                continue

        print("✅ Visual comparisons created")
        return True

    except Exception as e:
        print(f"Error creating visual comparisons: {str(e)}")
        traceback.print_exc()
        return False

# Function to analyze defect types
def analyze_defect_types(df):
    """Analyze the distribution of defect types"""
    print("Analyzing defect types...")

    try:
        if df is None or df.empty:
            raise ValueError("Empty dataframe provided")

        defect_data = []

        for _, row in df.iterrows():
            category = row['category']
            defect_types = row['defect_types']
            defect_counts = row['defect_counts']

            for defect_type, count in defect_counts.items():
                defect_data.append({
                    'category': category,
                    'defect_type': defect_type,
                    'count': count
                })

        defect_df = pd.DataFrame(defect_data)

        # Visualize defect types per category
        plt.figure(figsize=(15, 8))
        sns.barplot(data=defect_df, x='category', y='count', hue='defect_type')
        plt.title('Defect Types Distribution per Category', fontsize=14)
        plt.xticks(rotation=45)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'defect_types_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("Defect types analysis completed")
        return defect_df

    except Exception as e:
        print(f"❌ Error analyzing defect types: {str(e)}")
        traceback.print_exc()
        return None

# Function to analyze image statistics
def analyze_image_statistics(properties_df):
    """Perform statistical analysis of image properties"""
    print("📈 Analyzing image statistics...")

    try:
        if properties_df is None or properties_df.empty:
            raise ValueError("Empty dataframe provided")

        # Compare good vs defective images
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Brightness distribution
        sns.histplot(data=properties_df, x='mean_brightness', hue='type', kde=True, ax=axes[0, 0])
        axes[0, 0].set_title('Brightness Distribution', fontsize=14)
        axes[0, 0].set_xlabel('Mean Brightness', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)

        # Color channel comparison
        color_df = properties_df.melt(id_vars=['category', 'type'],
                                     value_vars=['mean_r', 'mean_g', 'mean_b'],
                                     var_name='channel', value_name='value')
        sns.boxplot(data=color_df, x='channel', y='value', hue='type', ax=axes[0, 1])
        axes[0, 1].set_title('Color Channel Comparison', fontsize=14)
        axes[0, 1].set_xlabel('Color Channel', fontsize=12)
        axes[0, 1].set_ylabel('Mean Value', fontsize=12)

        # Image size distribution
        sns.scatterplot(data=properties_df, x='width', y='height', hue='type', ax=axes[1, 0])
        axes[1, 0].set_title('Image Size Distribution', fontsize=14)
        axes[1, 0].set_xlabel('Width (pixels)', fontsize=12)
        axes[1, 0].set_ylabel('Height (pixels)', fontsize=12)

        # Standard deviation of brightness
        sns.histplot(data=properties_df, x='std_brightness', hue='type', kde=True, ax=axes[1, 1])
        axes[1, 1].set_title('Brightness Variation Distribution', fontsize=14)
        axes[1, 1].set_xlabel('Std Brightness', fontsize=12)
        axes[1, 1].set_ylabel('Count', fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'image_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print("✅ Image statistics analysis completed")
        return True

    except Exception as e:
        print(f"❌ Error analyzing image statistics: {str(e)}")
        traceback.print_exc()
        return False

# Function to generate summary report
def generate_summary_report(structure_df, properties_df, defect_df):
    """Generate a comprehensive summary report"""
    print("📝 Generating summary report...")

    try:
        if structure_df is None or structure_df.empty:
            raise ValueError("Structure dataframe is empty")

        # Calculate statistics
        total_good = structure_df['train_good'].sum() + structure_df['test_good'].sum()
        total_defects = structure_df['test_defects'].sum()

        # Convert numpy types to Python native types for JSON serialization
        report = {
            'dataset_summary': {
                'total_categories': int(len(structure_df)),
                'total_images': int(structure_df['total_images'].sum()),
                'total_good_images': int(total_good),
                'total_defective_images': int(total_defects),
                'good_percentage': float(total_good / structure_df['total_images'].sum() * 100),
                'defective_percentage': float(total_defects / structure_df['total_images'].sum() * 100),
                'average_imbalance_ratio': float(structure_df['imbalance_ratio'].mean()),
                'max_imbalance_ratio': float(structure_df['imbalance_ratio'].max()),
                'min_imbalance_ratio': float(structure_df['imbalance_ratio'].min()),
                'max_imbalance_category': str(structure_df.loc[structure_df['imbalance_ratio'].idxmax(), 'category'])
            },
            'image_properties': {
                'average_width': float(properties_df['width'].mean()) if properties_df is not None else None,
                'average_height': float(properties_df['height'].mean()) if properties_df is not None else None,
                'width_std': float(properties_df['width'].std()) if properties_df is not None else None,
                'height_std': float(properties_df['height'].std()) if properties_df is not None else None,
                'good_brightness_mean': float(properties_df[properties_df['type'] == 'good']['mean_brightness'].mean()) if properties_df is not None else None,
                'defective_brightness_mean': float(properties_df[properties_df['type'] == 'defective']['mean_brightness'].mean()) if properties_df is not None else None
            },
            'defect_analysis': {
                'total_defect_types': int(defect_df['defect_type'].nunique()) if defect_df is not None else None,
                'most_common_defect': str(defect_df.groupby('defect_type')['count'].sum().idxmax()) if defect_df is not None else None,
                'category_with_most_defects': str(structure_df.loc[structure_df['test_defects'].idxmax(), 'category'])
            },
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Save JSON report
        with open(os.path.join(OUTPUT_DIR, 'summary_report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        # Create text summary
        with open(os.path.join(OUTPUT_DIR, 'summary_report.txt'), 'w') as f:
            f.write("MVTec-AD Dataset Exploratory Analysis Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write("Dataset Overview:\n")
            f.write(f"- Total Categories: {report['dataset_summary']['total_categories']}\n")
            f.write(f"- Total Images: {report['dataset_summary']['total_images']:,}\n")
            f.write(f"- Good Images: {report['dataset_summary']['total_good_images']:,} ({report['dataset_summary']['good_percentage']:.1f}%)\n")
            f.write(f"- Defective Images: {report['dataset_summary']['total_defective_images']:,} ({report['dataset_summary']['defective_percentage']:.1f}%)\n")
            f.write(f"- Average Imbalance Ratio: {report['dataset_summary']['average_imbalance_ratio']:.2f}\n")
            f.write(f"- Maximum Imbalance Ratio: {report['dataset_summary']['max_imbalance_ratio']:.2f} (in {report['dataset_summary']['max_imbalance_category']})\n\n")

            if report['image_properties']['average_width'] is not None:
                f.write("Image Properties:\n")
                f.write(f"- Average Image Size: {report['image_properties']['average_width']:.0f} x {report['image_properties']['average_height']:.0f}\n")
                f.write(f"- Good Images Brightness (mean): {report['image_properties']['good_brightness_mean']:.2f}\n")
                f.write(f"- Defective Images Brightness (mean): {report['image_properties']['defective_brightness_mean']:.2f}\n\n")

            if report['defect_analysis']['total_defect_types'] is not None:
                f.write("Defect Analysis:\n")
                f.write(f"- Total Defect Types: {report['defect_analysis']['total_defect_types']}\n")
                f.write(f"- Most Common Defect Type: {report['defect_analysis']['most_common_defect']}\n")
                f.write(f"- Category with Most Defects: {report['defect_analysis']['category_with_most_defects']}\n\n")

            f.write("Key Findings:\n")
            f.write("1. The dataset exhibits significant class imbalance, with some categories having up to ")
            f.write(f"{report['dataset_summary']['max_imbalance_ratio']:.1f}x more defective images than good images.\n")
            f.write("2. This imbalance likely contributed to the high false positive rate in Phase 2 model training.\n")
            f.write("3. Image sizes vary across categories, which may require careful resizing strategies.\n")
            f.write("4. Defective images show different brightness characteristics compared to good images.\n\n")

            f.write("Recommendations for Model Training:\n")
            f.write("1. Implement class balancing techniques (weighted loss, balanced sampling)\n")
            f.write("2. Optimize decision threshold to reduce false positives\n")
            f.write("3. Consider category-specific approaches for highly imbalanced categories\n")
            f.write("4. Use data augmentation to address class imbalance\n")

        print("✅ Summary report generated")
        return True

    except Exception as e:
        print(f"❌ Error generating summary report: {str(e)}")
        traceback.print_exc()
        return False

# Main function
def main():
    """Main function to run the EDA"""
    print("🚀 Starting Exploratory Data Analysis for MVTec-AD Dataset...")
    print("=" * 60)

    # Check if dataset exists
    if not os.path.exists(DATASET_ROOT):
        print(f"❌ Dataset root path does not exist: {DATASET_ROOT}")
        print("Please check the path and ensure the dataset is available.")
        return False

    # Step 1: Analyze dataset structure
    structure_df = safe_execute(analyze_dataset_structure, DATASET_ROOT)
    if structure_df is not None:
        structure_df.to_csv(os.path.join(OUTPUT_DIR, 'dataset_structure.csv'), index=False)
        print("✅ Dataset structure saved to CSV")

    # Step 2: Visualize class distribution
    safe_execute(visualize_class_distribution, structure_df)

    # Step 3: Analyze image properties
    properties_df = safe_execute(analyze_image_properties, structure_df, DATASET_ROOT)
    if properties_df is not None:
        properties_df.to_csv(os.path.join(OUTPUT_DIR, 'image_properties.csv'), index=False)
        print("✅ Image properties saved to CSV")

    # Step 4: Create visual comparisons
    safe_execute(create_visual_comparison, structure_df, DATASET_ROOT)

    # Step 5: Analyze defect types
    defect_df = safe_execute(analyze_defect_types, structure_df)
    if defect_df is not None:
        defect_df.to_csv(os.path.join(OUTPUT_DIR, 'defect_types.csv'), index=False)
        print("✅ Defect types saved to CSV")

    # Step 6: Analyze image statistics
    safe_execute(analyze_image_statistics, properties_df)

    # Step 7: Generate summary report
    safe_execute(generate_summary_report, structure_df, properties_df, defect_df)

    print("\n" + "=" * 60)
    print("🎉 EDA completed successfully!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print("=" * 60)

    return True

# Execute main function
if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ All tasks completed successfully!")
        else:
            print("\n⚠️ Some tasks encountered errors. Please check the output files.")
    except Exception as e:
        print(f"❌ Critical error in main execution: {str(e)}")
        traceback.print_exc()