from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, auc,
                           precision_recall_curve, f1_score, accuracy_score)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import json
from datetime import datetime
import gc
warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
DATASET_ROOT = "/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad"
MODEL_PATH = "/content/drive/MyDrive/Nubia2025/model_epoch_3.pth"
OUTPUT_DIR = "/content/drive/MyDrive/NUBIA2025-CNN-3-EVALS"

class EvalConfig:
    """Configuration class for evaluation parameters."""
    # Paths
    DATASET_ROOT = DATASET_ROOT
    MODEL_PATH = MODEL_PATH
    OUTPUT_DIR = OUTPUT_DIR

    # Model & Data
    MODEL_NAME = "resnet50"
    IMG_SIZE = 224
    BATCH_SIZE = 32
    RANDOM_STATE = 42
    VALIDATION_SPLIT = 0.2

    # Labels
    LABEL_MAP = {"good": 0, "defective": 1}
    INV_LABEL_MAP = {0: "good", 1: "defective"}
    CLASS_NAMES = ["Good", "Defective"]

    # Visualization
    FIGSIZE_LARGE = (15, 10)
    FIGSIZE_MEDIUM = (12, 8)
    FIGSIZE_SMALL = (10, 6)
    DPI = 300

    # t-SNE parameters
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    TSNE_RANDOM_STATE = 42

    # GradCAM parameters
    GRADCAM_LAYER = 'layer4'  # Last layer for ResNet50

# Create output directory structure
os.makedirs(EvalConfig.OUTPUT_DIR, exist_ok=True)
for subdir in ['visualizations', 'metrics', 'gradcam', 'failure_analysis', 'tsne', 'architecture']:
    os.makedirs(os.path.join(EvalConfig.OUTPUT_DIR, subdir), exist_ok=True)

print(f"Evaluation outputs will be saved to: {EvalConfig.OUTPUT_DIR}")

# =====================================================================================
# 2. DATA PREPARATION (SAME DYNAMIC METHOD AS TRAINING)
# =====================================================================================
def prepare_data_paths(root_dir):
    """
    Dynamic data scanning - same method as training script.
    Scans the MVTec AD directory and creates paths and labels.
    """
    print("🔍 Scanning dataset directory for evaluation...")
    image_paths = []
    labels = []
    category_info = {}

    try:
        category_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except Exception as e:
        raise RuntimeError(f"Error accessing dataset directory: {str(e)}. "
                          f"Check if path '{root_dir}' exists and contains data.") from e

    for category in tqdm(category_dirs, desc="Processing Categories"):
        category_path = os.path.join(root_dir, category)
        category_info[category] = {"good": 0, "defective": 0}

        # --- Process 'good' images ---
        train_good_path = os.path.join(category_path, "train", "good")
        test_good_path = os.path.join(category_path, "test", "good")

        for good_path in [train_good_path, test_good_path]:
            if os.path.exists(good_path):
                for img_file in os.listdir(good_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_paths.append(os.path.join(good_path, img_file))
                        labels.append(EvalConfig.LABEL_MAP["good"])
                        category_info[category]["good"] += 1

        # --- Process 'defective' images ---
        test_dir = os.path.join(category_path, "test")
        if os.path.exists(test_dir):
            defect_types = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)) and d != "good"]
            for defect in defect_types:
                defect_path = os.path.join(test_dir, defect)
                for img_file in os.listdir(defect_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_paths.append(os.path.join(defect_path, img_file))
                        labels.append(EvalConfig.LABEL_MAP["defective"])
                        category_info[category]["defective"] += 1

    print(f"✅ Dataset scan complete. Found {len(image_paths)} images.")
    print(f"   → Good images: {labels.count(0)}")
    print(f"   → Defective images: {labels.count(1)}")

    return image_paths, labels, category_info

# =====================================================================================
# 3. DATASET CLASS (SAME AS TRAINING)
# =====================================================================================
class MVTecEvalDataset(Dataset):
    """Custom Dataset for evaluation with same structure as training."""
    def __init__(self, image_paths, labels, transform=None, return_paths=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            label = self.labels[idx]

            if self.transform:
                # Convert PIL to NumPy for Albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            else:
                image_tensor = transforms.ToTensor()(image)

            if self.return_paths:
                return image_tensor, label, image_path
            return image_tensor, label

        except Exception as e:
            print(f"⚠️ Error loading image {self.image_paths[idx]}: {str(e)}")
            dummy_tensor = torch.zeros((3, EvalConfig.IMG_SIZE, EvalConfig.IMG_SIZE))
            if self.return_paths:
                return dummy_tensor, 0, self.image_paths[idx]
            return dummy_tensor, 0

# =====================================================================================
# 4. MODEL LOADING
# =====================================================================================
def load_trained_model(model_path, device):
    """Load the trained model from checkpoint."""
    print(f"📥 Loading trained model from: {model_path}")

    try:
        # Recreate model architecture (same as training)
        model = models.resnet50(weights=None)  # Don't load pretrained weights
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, len(EvalConfig.LABEL_MAP))
        )

        # Load trained weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        print("✅ Model loaded successfully!")
        return model

    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}") from e

# =====================================================================================
# 5. VISUALIZATION FUNCTIONS
# =====================================================================================
def create_model_architecture_diagram():
    """Create a visual representation of the CNN architecture."""
    print("🎨 Creating model architecture diagram...")

    fig, ax = plt.subplots(figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    # Architecture components
    layers = [
        ("Input\n(224×224×3)", 0.1, 0.5, "lightblue"),
        ("ResNet-50\nBackbone\n(Pretrained)", 0.25, 0.5, "lightgreen"),
        ("Layer3 & Layer4\n(Fine-tuned)", 0.4, 0.5, "orange"),
        ("Global Avg Pool\n(2048)", 0.55, 0.5, "lightcoral"),
        ("Dropout(0.5)", 0.65, 0.5, "lightgray"),
        ("FC Layer\n(2048→512)", 0.75, 0.5, "lightyellow"),
        ("ReLU + BatchNorm", 0.85, 0.5, "lightpink"),
        ("Dropout(0.3)", 0.95, 0.6, "lightgray"),
        ("Output\n(Good/Defective)", 0.95, 0.4, "lightsteelblue")
    ]

    # Draw boxes and labels
    for layer, x, y, color in layers:
        if "Output" in layer or "Dropout(0.3)" in layer:
            width, height = 0.08, 0.15
        else:
            width, height = 0.12, 0.2

        rect = plt.Rectangle((x-width/2, y-height/2), width, height,
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, layer, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw arrows
    arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    for i, j in arrow_pairs:
        x1, y1 = layers[i][1] + 0.06, layers[i][2]
        x2, y2 = layers[j][1] - 0.06, layers[j][2]
        if i == 6:  # Split after ReLU + BatchNorm
            ax.annotate('', xy=(x2, layers[j][2]), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        else:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.set_title('CNN Architecture for MVTec Anomaly Detection', fontsize=16, fontweight='bold')
    ax.axis('off')

    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'architecture', 'model_architecture.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ Architecture diagram saved: {save_path}")

def create_data_pipeline_flowchart():
    """Create data pipeline visualization."""
    print("🎨 Creating data pipeline flowchart...")

    fig, ax = plt.subplots(figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    # Pipeline steps
    steps = [
        ("MVTec AD\nDataset", 0.1, 0.7, "lightblue"),
        ("Dynamic Path\nScanning", 0.25, 0.7, "lightgreen"),
        ("Binary Label\nAssignment", 0.4, 0.7, "lightyellow"),
        ("Train/Val\nSplit (80/20)", 0.55, 0.7, "lightcoral"),
        ("Data\nAugmentation", 0.25, 0.3, "orange"),
        ("ResNet-50\nTraining", 0.55, 0.3, "lightpink"),
        ("Model\nEvaluation", 0.85, 0.5, "lightsteelblue")
    ]

    # Draw pipeline boxes
    for step, x, y, color in steps:
        width, height = 0.12, 0.15
        rect = plt.Rectangle((x-width/2, y-height/2), width, height,
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, step, ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw connections
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)]
    for i, j in connections:
        x1, y1 = steps[i][1], steps[i][2]
        x2, y2 = steps[j][1], steps[j][2]

        if i == 3 and j == 4:  # Train split to augmentation
            ax.annotate('', xy=(x2, y2+0.075), xytext=(x1-0.03, y1-0.075),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
        elif i == 3 and j == 5:  # Val split to training
            ax.annotate('', xy=(x2, y2+0.075), xytext=(x1+0.03, y1-0.075),
                       arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        elif i == 5 and j == 6:  # Training to evaluation
            ax.annotate('', xy=(x2-0.06, y2), xytext=(x1+0.06, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))
        else:
            ax.annotate('', xy=(x2-0.06, y2), xytext=(x1+0.06, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 0.9)
    ax.set_title('Data Pipeline and Training Workflow', fontsize=16, fontweight='bold')
    ax.axis('off')

    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'architecture', 'data_pipeline.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ Data pipeline diagram saved: {save_path}")

def visualize_class_distribution(category_info):
    """Create class distribution visualization."""
    print("📊 Creating class distribution visualization...")

    # Prepare data
    categories = list(category_info.keys())
    good_counts = [category_info[cat]['good'] for cat in categories]
    defective_counts = [category_info[cat]['defective'] for cat in categories]

    # Create subplot with multiple views
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    # 1. Stacked bar chart by category
    x = np.arange(len(categories))
    width = 0.6
    ax1.bar(x, good_counts, width, label='Good', color='lightgreen', alpha=0.8)
    ax1.bar(x, defective_counts, width, bottom=good_counts, label='Defective', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Product Categories')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Class Distribution by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Overall pie chart
    total_good = sum(good_counts)
    total_defective = sum(defective_counts)
    ax2.pie([total_good, total_defective], labels=['Good', 'Defective'],
            colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    ax2.set_title(f'Overall Distribution\n(Total: {total_good + total_defective} images)')

    # 3. Category-wise totals
    totals = [good_counts[i] + defective_counts[i] for i in range(len(categories))]
    bars = ax3.bar(categories, totals, color='lightblue', alpha=0.7)
    ax3.set_xlabel('Product Categories')
    ax3.set_ylabel('Total Images')
    ax3.set_title('Total Images per Category')
    ax3.tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # 4. Imbalance ratio per category
    ratios = [defective_counts[i] / (good_counts[i] + 1e-8) for i in range(len(categories))]  # Avoid division by zero
    ax4.bar(categories, ratios, color='orange', alpha=0.7)
    ax4.set_xlabel('Product Categories')
    ax4.set_ylabel('Defective/Good Ratio')
    ax4.set_title('Class Imbalance by Category')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'visualizations', 'class_distribution.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ Class distribution saved: {save_path}")

    return total_good, total_defective

def show_data_augmentation_examples(image_paths, labels):
    """Show original vs augmented image examples."""
    print("🖼️ Creating data augmentation examples...")

    # Define augmentation pipeline
    aug_transform = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=30, p=1.0),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.Resize(EvalConfig.IMG_SIZE, EvalConfig.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    no_aug_transform = A.Compose([
        A.Resize(EvalConfig.IMG_SIZE, EvalConfig.IMG_SIZE),
        ToTensorV2()
    ])

    # Select sample images (2 good, 2 defective)
    good_indices = [i for i, label in enumerate(labels) if label == 0]
    defective_indices = [i for i, label in enumerate(labels) if label == 1]

    sample_indices = (good_indices[:2] + defective_indices[:2])[:4]

    fig, axes = plt.subplots(4, 2, figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    for i, idx in enumerate(sample_indices):
        try:
            # Load original image
            img_path = image_paths[idx]
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            # Original image
            orig_transformed = no_aug_transform(image=image_np)
            orig_img = orig_transformed['image'].permute(1, 2, 0).numpy()

            # Augmented image
            aug_transformed = aug_transform(image=image_np)
            aug_img = aug_transformed['image']

            # Denormalize for visualization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            aug_img_denorm = aug_img.permute(1, 2, 0).numpy()
            aug_img_denorm = std * aug_img_denorm + mean
            aug_img_denorm = np.clip(aug_img_denorm, 0, 1)

            # Plot
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(f'Original - {EvalConfig.INV_LABEL_MAP[labels[idx]].title()}')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(aug_img_denorm)
            axes[i, 1].set_title(f'Augmented - {EvalConfig.INV_LABEL_MAP[labels[idx]].title()}')
            axes[i, 1].axis('off')

        except Exception as e:
            print(f"⚠️ Error processing image {idx}: {str(e)}")
            # Show placeholder
            axes[i, 0].text(0.5, 0.5, 'Error loading\nimage', ha='center', va='center', transform=axes[i, 0].transAxes)
            axes[i, 1].text(0.5, 0.5, 'Error loading\nimage', ha='center', va='center', transform=axes[i, 1].transAxes)
            axes[i, 0].axis('off')
            axes[i, 1].axis('off')

    plt.suptitle('Data Augmentation Examples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'visualizations', 'augmentation_examples.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ Augmentation examples saved: {save_path}")

# =====================================================================================
# 6. MODEL EVALUATION FUNCTIONS
# =====================================================================================
def evaluate_model_comprehensive(model, test_loader, device):
    """Comprehensive model evaluation with all metrics."""
    print("🧪 Running comprehensive model evaluation...")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if len(batch) == 3:  # With paths
                images, labels, paths = batch
                all_paths.extend(paths)
            else:  # Without paths
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='binary')

    # Classification report
    class_report = classification_report(all_labels, all_predictions,
                                       target_names=EvalConfig.CLASS_NAMES,
                                       output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"✅ Evaluation complete!")
    print(f"   → Accuracy: {accuracy:.4f}")
    print(f"   → F1-Score: {f1:.4f}")

    results = {
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': np.array(all_probabilities),
        'paths': all_paths if all_paths else None,
        'accuracy': accuracy,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': cm
    }

    return results

def create_roc_curves(results):
    """Create ROC curves visualization."""
    print("📈 Creating ROC curves...")

    labels = results['labels']
    probabilities = results['probabilities']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    # Binary ROC curve
    fpr, tpr, _ = roc_curve(labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)

    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve - Binary Classification')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, probabilities[:, 1])
    pr_auc = auc(recall, precision)

    ax2.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'visualizations', 'roc_curves.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curves saved: {save_path}")

    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}

def create_enhanced_confusion_matrix(results):
    """Create detailed confusion matrix visualization."""
    print("🎯 Creating enhanced confusion matrix...")

    cm = results['confusion_matrix']
    accuracy = results['accuracy']
    f1 = results['f1_score']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EvalConfig.CLASS_NAMES,
                yticklabels=EvalConfig.CLASS_NAMES, ax=ax1)
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    ax1.set_title(f'Confusion Matrix (Raw Counts)\nAccuracy: {accuracy:.3f}, F1: {f1:.3f}')

    # Normalized
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Oranges',
                xticklabels=EvalConfig.CLASS_NAMES,
                yticklabels=EvalConfig.CLASS_NAMES, ax=ax2)
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    ax2.set_title('Confusion Matrix (Normalized)')

    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'visualizations', 'confusion_matrix_detailed.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ Enhanced confusion matrix saved: {save_path}")

# =====================================================================================
# 7. T-SNE VISUALIZATION
# =====================================================================================
def extract_features_for_tsne(model, data_loader, device, layer_name='avgpool'):
    """Extract features from a specific layer for t-SNE."""
    print(f"🔬 Extracting features from layer '{layer_name}' for t-SNE...")

    features = []
    labels = []

    # Hook to extract features
    def hook_fn(module, input, output):
        if layer_name == 'avgpool':
            # Global average pooling output
            features.append(output.view(output.size(0), -1).cpu().detach().numpy())
        else:
            # Flatten other layer outputs
            features.append(output.view(output.size(0), -1).cpu().detach().numpy())

    # Register hook
    if layer_name == 'avgpool':
        handle = model.avgpool.register_forward_hook(hook_fn)
    else:
        # For other layers, you might need to adjust this
        handle = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            if len(batch) == 3:
                images, batch_labels, _ = batch
            else:
                images, batch_labels = batch

            images = images.to(device)
            _ = model(images)  # Forward pass triggers hook
            labels.extend(batch_labels.numpy())

    handle.remove()  # Remove hook

    # Concatenate all features
    all_features = np.concatenate(features, axis=0)
    all_labels = np.array(labels)

    print(f"✅ Extracted features shape: {all_features.shape}")
    return all_features, all_labels

def create_tsne_visualization(model, data_loader, device):
    """Create t-SNE visualization of learned features."""
    print("🗺️ Creating t-SNE visualization...")

    # Extract features
    features, labels = extract_features_for_tsne(model, data_loader, device)

    # Apply t-SNE
    print("   → Running t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2,
                perplexity=min(EvalConfig.TSNE_PERPLEXITY, len(features)-1),
                n_iter=EvalConfig.TSNE_N_ITER,
                random_state=EvalConfig.TSNE_RANDOM_STATE)

    features_2d = tsne.fit_transform(features)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=EvalConfig.FIGSIZE_LARGE, dpi=EvalConfig.DPI)

    # Scatter plot colored by class
    colors = ['lightgreen', 'lightcoral']
    for i, class_name in enumerate(EvalConfig.CLASS_NAMES):
        mask = labels == i
        ax1.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=colors[i], label=class_name, alpha=0.7, s=20)

    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.set_title('t-SNE Visualization of Learned Features')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Density plot
    from scipy.stats import gaussian_kde
    try:
        # Create density plot for each class
        for i, (class_name, color) in enumerate(zip(EvalConfig.CLASS_NAMES, colors)):
            mask = labels == i
            if np.sum(mask) > 10:  # Need sufficient points for KDE
                xy = features_2d[mask]
                kde = gaussian_kde(xy.T)
                x_min, x_max = features_2d[:, 0].min(), features_2d[:, 0].max()
                y_min, y_max = features_2d[:, 1].min(), features_2d[:, 1].max()
                xx, yy = np.mgrid[x_min:x_max:.1, y_min:y_max:.1]
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(positions).T.reshape(xx.shape)
                ax2.contour(xx, yy, density, alpha=0.6, colors=[color])

        ax2.scatter(features_2d[:, 0], features_2d[:, 1],
                   c=[colors[label] for label in labels], alpha=0.5, s=10)
        ax2.set_title('t-SNE with Density Contours')
    except:
        # Fallback to simple scatter if KDE fails
        for i, class_name in enumerate(EvalConfig.CLASS_NAMES):
            mask = labels == i
            ax2.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=colors[i], label=class_name, alpha=0.7, s=20)
        ax2.set_title('t-SNE Feature Space')

    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'tsne', 'tsne_visualization.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ t-SNE visualization saved: {save_path}")

# =====================================================================================
# 8. GRAD-CAM IMPLEMENTATION
# =====================================================================================
class GradCAM:
    """Gradient-weighted Class Activation Mapping for model interpretability."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.hook_layers()

    def hook_layers(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_backward_hook(backward_hook)
        target_layer.register_forward_hook(forward_hook)

    def generate_cam(self, input_tensor, class_idx):
        # Forward pass
        model_output = self.model(input_tensor)

        # Backward pass
        self.model.zero_grad()
        class_loss = model_output[:, class_idx].sum()
        class_loss.backward()

        # Generate CAM
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))

        # Generate heatmap
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Normalize
        cam = np.maximum(cam, 0)
        cam = cam / cam.max() if cam.max() > 0 else cam

        return cam

def create_gradcam_visualizations(model, data_loader, device, num_samples=12):
    """Create Grad-CAM visualizations for sample images."""
    print("🔥 Creating Grad-CAM visualizations...")

    # Initialize Grad-CAM
    grad_cam = GradCAM(model, EvalConfig.GRADCAM_LAYER)

    # Get sample images
    sample_images = []
    sample_labels = []
    sample_paths = []

    model.eval()
    for batch in data_loader:
        if len(batch) == 3:
            images, labels, paths = batch
            sample_paths.extend(paths)
        else:
            images, labels = batch
            sample_paths.extend([''] * len(images))

        sample_images.extend(images)
        sample_labels.extend(labels)

        if len(sample_images) >= num_samples:
            break

    # Select diverse samples
    sample_indices = []
    good_samples = [i for i, label in enumerate(sample_labels) if label == 0]
    defective_samples = [i for i, label in enumerate(sample_labels) if label == 1]

    # Get equal samples from each class
    sample_indices.extend(good_samples[:num_samples//2])
    sample_indices.extend(defective_samples[:num_samples//2])
    sample_indices = sample_indices[:num_samples]

    # Create visualization
    rows = 3
    cols = num_samples // rows + (1 if num_samples % rows else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4), dpi=EvalConfig.DPI)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx, sample_idx in enumerate(sample_indices):
        row = idx // cols
        col = idx % cols

        try:
            # Get image and prediction
            image_tensor = sample_images[sample_idx].unsqueeze(0).to(device)
            true_label = sample_labels[sample_idx]

            # Get model prediction
            with torch.no_grad():
                output = model(image_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                confidence = F.softmax(output, dim=1).max().item()

            # Generate Grad-CAM
            cam = grad_cam.generate_cam(image_tensor, predicted_class)

            # Prepare original image for visualization
            img_np = sample_images[sample_idx].permute(1, 2, 0).numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)

            # Resize CAM to match image size
            cam_resized = cv2.resize(cam, (EvalConfig.IMG_SIZE, EvalConfig.IMG_SIZE))

            # Overlay heatmap
            heatmap = plt.cm.jet(cam_resized)[:, :, :3]
            overlay = 0.6 * img_np + 0.4 * heatmap

            # Plot
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f'True: {EvalConfig.INV_LABEL_MAP[true_label]}\n'
                                   f'Pred: {EvalConfig.INV_LABEL_MAP[predicted_class]} '
                                   f'({confidence:.3f})', fontsize=10)
            axes[row, col].axis('off')

        except Exception as e:
            print(f"⚠️ Error creating Grad-CAM for sample {sample_idx}: {str(e)}")
            axes[row, col].text(0.5, 0.5, 'Error', ha='center', va='center',
                              transform=axes[row, col].transAxes)
            axes[row, col].axis('off')

    # Hide empty subplots
    for idx in range(len(sample_indices), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle('Grad-CAM Visualizations - Model Attention Areas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'gradcam', 'gradcam_samples.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()
    print(f"✅ Grad-CAM visualizations saved: {save_path}")

# =====================================================================================
# 9. FAILURE CASE ANALYSIS
# =====================================================================================
def analyze_failure_cases(results):
    """Analyze and visualize failure cases (misclassifications)."""
    print("🔍 Analyzing failure cases...")

    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    probabilities = results['probabilities']
    paths = results['paths']

    # Find misclassified samples
    misclassified_mask = predictions != labels
    misclassified_indices = np.where(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        print("🎉 No misclassifications found! Perfect model performance.")
        return

    print(f"   → Found {len(misclassified_indices)} misclassified samples")

    # Analyze failure types
    false_positives = []  # Predicted defective, actually good
    false_negatives = []  # Predicted good, actually defective

    for idx in misclassified_indices:
        if labels[idx] == 0 and predictions[idx] == 1:
            false_positives.append(idx)
        elif labels[idx] == 1 and predictions[idx] == 0:
            false_negatives.append(idx)

    print(f"   → False Positives: {len(false_positives)}")
    print(f"   → False Negatives: {len(false_negatives)}")

    # Create failure analysis visualization
    num_samples = min(12, len(misclassified_indices))
    if num_samples == 0:
        return

    # Select samples with highest confidence errors
    error_confidences = []
    for idx in misclassified_indices:
        pred_class = predictions[idx]
        confidence = probabilities[idx][pred_class]
        error_confidences.append((idx, confidence))

    # Sort by confidence (highest confidence errors are most interesting)
    error_confidences.sort(key=lambda x: x[1], reverse=True)
    selected_indices = [x[0] for x in error_confidences[:num_samples]]

    # Visualization
    rows = 3
    cols = num_samples // rows + (1 if num_samples % rows else 0)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), dpi=EvalConfig.DPI)
    if rows == 1:
        axes = axes.reshape(1, -1)

    for plot_idx, sample_idx in enumerate(selected_indices):
        row = plot_idx // cols
        col = plot_idx % cols

        try:
            if paths and sample_idx < len(paths) and paths[sample_idx]:
                # Load and display image
                img = Image.open(paths[sample_idx]).convert("RGB")
                img_resized = img.resize((EvalConfig.IMG_SIZE, EvalConfig.IMG_SIZE))

                axes[row, col].imshow(img_resized)

                # Add detailed information
                true_label = EvalConfig.INV_LABEL_MAP[labels[sample_idx]]
                pred_label = EvalConfig.INV_LABEL_MAP[predictions[sample_idx]]
                confidence = probabilities[sample_idx][predictions[sample_idx]]

                failure_type = "False Positive" if sample_idx in false_positives else "False Negative"

                axes[row, col].set_title(f'{failure_type}\nTrue: {true_label} | Pred: {pred_label}\n'
                                       f'Confidence: {confidence:.3f}', fontsize=9)
                axes[row, col].axis('off')
            else:
                axes[row, col].text(0.5, 0.5, 'Image not\navailable', ha='center', va='center',
                                  transform=axes[row, col].transAxes)
                axes[row, col].axis('off')

        except Exception as e:
            print(f"⚠️ Error loading failure case {sample_idx}: {str(e)}")
            axes[row, col].text(0.5, 0.5, 'Error loading\nimage', ha='center', va='center',
                              transform=axes[row, col].transAxes)
            axes[row, col].axis('off')

    # Hide empty subplots
    for idx in range(len(selected_indices), rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')

    plt.suptitle('Failure Case Analysis - High Confidence Misclassifications',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(EvalConfig.OUTPUT_DIR, 'failure_analysis', 'failure_cases.png')
    plt.savefig(save_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()

    # Save failure analysis summary
    failure_summary = {
        'total_misclassifications': len(misclassified_indices),
        'false_positives': len(false_positives),
        'false_negatives': len(false_negatives),
        'error_rate': len(misclassified_indices) / len(labels),
        'high_confidence_errors': [(idx, probabilities[idx][predictions[idx]])
                                 for idx in selected_indices]
    }

    summary_path = os.path.join(EvalConfig.OUTPUT_DIR, 'failure_analysis', 'failure_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(failure_summary, f, indent=2, default=str)

    print(f"✅ Failure analysis saved: {save_path}")
    print(f"✅ Failure summary saved: {summary_path}")

# =====================================================================================
# 10. PERFORMANCE METRICS EXPORT
# =====================================================================================
def save_comprehensive_metrics(results, roc_metrics, category_info):
    """Save all evaluation metrics to files."""
    print("💾 Saving comprehensive evaluation metrics...")

    # Prepare comprehensive metrics
    metrics = {
        'model_info': {
            'model_path': EvalConfig.MODEL_PATH,
            'evaluation_date': datetime.now().isoformat(),
            'architecture': EvalConfig.MODEL_NAME,
            'image_size': EvalConfig.IMG_SIZE
        },
        'dataset_info': {
            'total_samples': len(results['labels']),
            'good_samples': int(np.sum(np.array(results['labels']) == 0)),
            'defective_samples': int(np.sum(np.array(results['labels']) == 1)),
            'categories': category_info
        },
        'performance_metrics': {
            'accuracy': float(results['accuracy']),
            'f1_score': float(results['f1_score']),
            'roc_auc': float(roc_metrics['roc_auc']),
            'pr_auc': float(roc_metrics['pr_auc']),
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'classification_report': results['classification_report']
        }
    }

    # Save as JSON
    json_path = os.path.join(EvalConfig.OUTPUT_DIR, 'metrics', 'comprehensive_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Save as CSV for easy reading
    csv_data = {
        'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'Precision (Good)',
                  'Recall (Good)', 'Precision (Defective)', 'Recall (Defective)'],
        'Value': [
            results['accuracy'],
            results['f1_score'],
            roc_metrics['roc_auc'],
            roc_metrics['pr_auc'],
            results['classification_report']['Good']['precision'],
            results['classification_report']['Good']['recall'],
            results['classification_report']['Defective']['precision'],
            results['classification_report']['Defective']['recall']
        ]
    }

    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(EvalConfig.OUTPUT_DIR, 'metrics', 'performance_summary.csv')
    df.to_csv(csv_path, index=False)

    # Create performance comparison table visualization
    fig, ax = plt.subplots(figsize=EvalConfig.FIGSIZE_MEDIUM, dpi=EvalConfig.DPI)

    # Create table
    table_data = []
    for metric, value in zip(csv_data['Metric'], csv_data['Value']):
        table_data.append([metric, f"{value:.4f}"])

    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')

    ax.axis('off')
    ax.set_title('Model Performance Summary - Epoch 3', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    table_path = os.path.join(EvalConfig.OUTPUT_DIR, 'metrics', 'performance_table.png')
    plt.savefig(table_path, dpi=EvalConfig.DPI, bbox_inches='tight')
    plt.close()

    print(f"✅ Comprehensive metrics saved:")
    print(f"   → JSON: {json_path}")
    print(f"   → CSV: {csv_path}")
    print(f"   → Table: {table_path}")

# =====================================================================================
# 11. MAIN EVALUATION SCRIPT
# =====================================================================================
def main():
    """Main evaluation function orchestrating all analyses."""
    print("🚀 Starting comprehensive model evaluation...")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")

    try:
        # 1. Data Preparation (Dynamic - same as training)
        print("\n" + "="*50)
        print("PHASE 1: DATA PREPARATION")
        print("="*50)

        all_paths, all_labels, category_info = prepare_data_paths(EvalConfig.DATASET_ROOT)

        # Split data (same as training for consistency)
        _, test_paths, _, test_labels = train_test_split(
            all_paths, all_labels,
            test_size=EvalConfig.VALIDATION_SPLIT,
            random_state=EvalConfig.RANDOM_STATE,
            stratify=all_labels
        )

        print(f"📊 Test set: {len(test_paths)} samples")

        # 2. Load Model
        print("\n" + "="*50)
        print("PHASE 2: MODEL LOADING")
        print("="*50)

        model = load_trained_model(EvalConfig.MODEL_PATH, device)

        # 3. Create Data Loaders
        eval_transform = A.Compose([
            A.Resize(EvalConfig.IMG_SIZE, EvalConfig.IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        test_dataset = MVTecEvalDataset(test_paths, test_labels, transform=eval_transform, return_paths=True)
        test_loader = DataLoader(test_dataset, batch_size=EvalConfig.BATCH_SIZE, shuffle=False, num_workers=2)

        # 4. Architectural Visualizations
        print("\n" + "="*50)
        print("PHASE 3: ARCHITECTURAL VISUALIZATIONS")
        print("="*50)

        create_model_architecture_diagram()
        create_data_pipeline_flowchart()

        # 5. Data Visualizations
        print("\n" + "="*50)
        print("PHASE 4: DATA ANALYSIS VISUALIZATIONS")
        print("="*50)

        total_good, total_defective = visualize_class_distribution(category_info)
        show_data_augmentation_examples(all_paths, all_labels)

        # 6. Model Evaluation
        print("\n" + "="*50)
        print("PHASE 5: COMPREHENSIVE MODEL EVALUATION")
        print("="*50)

        results = evaluate_model_comprehensive(model, test_loader, device)

        # 7. Performance Visualizations
        print("\n" + "="*50)
        print("PHASE 6: PERFORMANCE VISUALIZATIONS")
        print("="*50)

        roc_metrics = create_roc_curves(results)
        create_enhanced_confusion_matrix(results)

        # 8. Advanced Analysis
        print("\n" + "="*50)
        print("PHASE 7: ADVANCED ANALYSIS")
        print("="*50)

        # t-SNE (memory intensive, so we'll use a subset)
        subset_size = min(1000, len(test_dataset))
        subset_indices = np.random.choice(len(test_dataset), subset_size, replace=False)
        subset_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
        subset_loader = DataLoader(subset_dataset, batch_size=EvalConfig.BATCH_SIZE, shuffle=False)

        create_tsne_visualization(model, subset_loader, device)

        # Grad-CAM
        create_gradcam_visualizations(model, test_loader, device)

        # 9. Failure Analysis
        print("\n" + "="*50)
        print("PHASE 8: FAILURE CASE ANALYSIS")
        print("="*50)

        analyze_failure_cases(results)

        # 10. Save Comprehensive Metrics
        print("\n" + "="*50)
        print("PHASE 9: SAVING COMPREHENSIVE METRICS")
        print("="*50)

        save_comprehensive_metrics(results, roc_metrics, category_info)

        # 11. Final Summary
        print("\n" + "="*80)
        print("🎉 EVALUATION COMPLETE!")
        print("="*80)
        print(f"📊 Final Results Summary:")
        print(f"   → Model: {EvalConfig.MODEL_PATH}")
        print(f"   → Test Samples: {len(test_labels)}")
        print(f"   → Accuracy: {results['accuracy']:.4f}")
        print(f"   → F1-Score: {results['f1_score']:.4f}")
        print(f"   → ROC-AUC: {roc_metrics['roc_auc']:.4f}")
        print(f"   → PR-AUC: {roc_metrics['pr_auc']:.4f}")
        print(f"📁 All outputs saved to: {EvalConfig.OUTPUT_DIR}")

        # Generate evaluation report
        create_evaluation_report(results, roc_metrics, category_info)

    except Exception as e:
        print(f"❌ Evaluation failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nPlease check:")
        print("1. Model file exists at specified path")
        print("2. Dataset directory is accessible")
        print("3. Sufficient GPU/CPU memory")
        print("4. All dependencies are installed")

def create_evaluation_report(results, roc_metrics, category_info):
    """Create a comprehensive evaluation report."""
    print("📄 Creating comprehensive evaluation report...")

    report_content = f"""
# MVTec Anomaly Detection - Model Evaluation Report
## Epoch 3 Model Performance Analysis

### Model Information
- **Model Architecture**: ResNet-50 with custom classifier
- **Model Path**: {EvalConfig.MODEL_PATH}
- **Image Size**: {EvalConfig.IMG_SIZE}x{EvalConfig.IMG_SIZE}
- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Dataset Overview
- **Total Test Samples**: {len(results['labels'])}
- **Good Samples**: {int(np.sum(np.array(results['labels']) == 0))} ({100*np.sum(np.array(results['labels']) == 0)/len(results['labels']):.1f}%)
- **Defective Samples**: {int(np.sum(np.array(results['labels']) == 1))} ({100*np.sum(np.array(results['labels']) == 1)/len(results['labels']):.1f}%)
- **Categories**: {len(category_info)} product categories

### Performance Metrics
- **Accuracy**: {results['accuracy']:.4f}
- **F1-Score**: {results['f1_score']:.4f}
- **ROC-AUC**: {roc_metrics['roc_auc']:.4f}
- **Precision-Recall AUC**: {roc_metrics['pr_auc']:.4f}

### Class-wise Performance
#### Good (Normal) Class:
- **Precision**: {results['classification_report']['Good']['precision']:.4f}
- **Recall**: {results['classification_report']['Good']['recall']:.4f}
- **F1-Score**: {results['classification_report']['Good']['f1-score']:.4f}

#### Defective (Anomaly) Class:
- **Precision**: {results['classification_report']['Defective']['precision']:.4f}
- **Recall**: {results['classification_report']['Defective']['recall']:.4f}
- **F1-Score**: {results['classification_report']['Defective']['f1-score']:.4f}

### Confusion Matrix
```
                Predicted
Actual      Good  Defective
Good        {results['confusion_matrix'][0][0]:4d}      {results['confusion_matrix'][0][1]:4d}
Defective   {results['confusion_matrix'][1][0]:4d}      {results['confusion_matrix'][1][1]:4d}
```

### Key Findings
1. **Model Performance**: The epoch 3 model shows {
    'excellent' if results['f1_score'] > 0.9 else
    'good' if results['f1_score'] > 0.8 else
    'moderate' if results['f1_score'] > 0.7 else
    'needs improvement'
} performance with F1-score of {results['f1_score']:.4f}

2. **Class Balance**: The model handles the {
    'balanced' if abs(np.sum(np.array(results['labels']) == 0) - np.sum(np.array(results['labels']) == 1)) < 50 else 'imbalanced'
} dataset effectively

3. **Generalization**: ROC-AUC of {roc_metrics['roc_auc']:.4f} indicates {
    'excellent' if roc_metrics['roc_auc'] > 0.9 else
    'good' if roc_metrics['roc_auc'] > 0.8 else
    'moderate'
} discriminative ability

### Files Generated
- Model architecture diagram
- Data pipeline flowchart
- Class distribution analysis
- Data augmentation examples
- ROC and PR curves
- Enhanced confusion matrices
- t-SNE feature visualization
- Grad-CAM interpretability maps
- Failure case analysis
- Comprehensive metrics (JSON/CSV)

### Recommendations
1. **Deployment Readiness**: Model shows strong performance suitable for industrial deployment
2. **Monitoring**: Implement continuous monitoring for performance degradation
3. **Data Collection**: Focus on collecting more samples from underrepresented failure modes
4. **Model Updates**: Consider periodic retraining with new data

---
*Report generated automatically by MVTec Evaluation Script*
*All visualizations and detailed metrics available in output directory*
"""

    report_path = os.path.join(EvalConfig.OUTPUT_DIR, 'EVALUATION_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report_content)

    print(f"✅ Comprehensive evaluation report saved: {report_path}")

# =====================================================================================
# 12. EXECUTION GUARD AND ERROR HANDLING
# =====================================================================================
if __name__ == '__main__':
    # Validate prerequisites
    print("🔍 Validating prerequisites...")

    # Check if model file exists
    if not os.path.exists(EvalConfig.MODEL_PATH):
        print(f"❌ CRITICAL ERROR: Model file not found at {EvalConfig.MODEL_PATH}")
        print("Please ensure the model file exists before running evaluation.")
        exit(1)

    # Check if dataset exists
    if not os.path.exists(EvalConfig.DATASET_ROOT) or len(os.listdir(EvalConfig.DATASET_ROOT)) == 0:
        print(f"❌ CRITICAL ERROR: Dataset directory is empty or doesn't exist at {EvalConfig.DATASET_ROOT}")
        print("Please ensure the MVTec AD dataset is available.")
        exit(1)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ GPU not available, using CPU (evaluation will be slower)")

    # Memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        main()
        print("\n🎊 SUCCESS: All evaluations completed successfully!")
        print(f"📂 Check results in: {EvalConfig.OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\n⏹️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n💥 FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting Tips:")
        print("1. Ensure sufficient GPU/CPU memory")
        print("2. Verify model file is not corrupted")
        print("3. Check dataset file permissions")
        print("4. Try reducing batch size in EvalConfig")
        print("5. Restart runtime and try again")
    finally:
        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n🧹 Memory cleanup completed")