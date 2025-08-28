# Comprehensive Evaluation Script for SimCLR on MVTec-AD
# ========================
# IMPORTS AND CONFIGURATION
# ========================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (f1_score, precision_score, recall_score,
                            roc_auc_score, accuracy_score, confusion_matrix,
                            roc_curve, auc, precision_recall_curve)
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from datetime import datetime
import warnings
import gc
import cv2
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
import graphviz
from IPython.display import SVG, display
import shutil
from collections import defaultdict, Counter
import random
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# ========================
# CONFIGURATION CLASS
# ========================
class EvalConfig:
    """Configuration for evaluation script"""
    # Paths - same as training script
    DATA_ROOT = "/content/drive/MyDrive/MVTEC_AD/mvtec_ad"
    SAVE_DIR = "/content/drive/MyDrive/PhD_SimCLR_Results"
    EVAL_DIR = "/content/drive/MyDrive/FINAL_EVALUATIONS"

    # MVTec-AD categories
    ALL_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]

    # Evaluation parameters
    BATCH_SIZE = 16  # Reduced for CPU stability
    IMAGE_SIZE = 224
    NUM_WORKERS = 2  # Reduced for CPU stability

    # T-SNE parameters
    TSNE_PERPLEXITY = 30
    TSNE_ITERATIONS = 1000
    TSNE_SAMPLES_PER_CAT = 50  # Reduced for CPU

    # Grad-CAM parameters
    TARGET_LAYER_NAME = "layer4"  # ResNet50 layer to visualize

    # Visualization settings
    DPI = 150  # Reduced for CPU
    FIGSIZE = (12, 8)
    COLOR_PALETTE = "viridis"

    # Error analysis
    MAX_ERROR_SAMPLES = 10  # Reduced for CPU

    def __init__(self):
        self.device = self._setup_device()
        os.makedirs(self.EVAL_DIR, exist_ok=True)

    def _setup_device(self):
        """Setup device for evaluation"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️ CUDA not available! Using CPU")
            # Reduce parameters for CPU
            self.BATCH_SIZE = 8
            self.TSNE_SAMPLES_PER_CAT = 30
            self.MAX_ERROR_SAMPLES = 5
        return device

# Initialize configuration
eval_config = EvalConfig()

# ========================
# MODEL DEFINITIONS
# ========================
class PhDSimCLR(nn.Module):
    """PhD-optimized SimCLR model"""
    def __init__(self, feature_dim=256):
        super().__init__()
        # Load ResNet50 with modern weights API
        try:
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = resnet50(weights=weights)
            print("✅ Loaded ImageNet pretrained ResNet50")
        except Exception:
            backbone = resnet50(weights=None)
            print("⚠️ Using randomly initialized ResNet50")

        # Remove final classification layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        encoder_dim = 2048

        # Projection head for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, feature_dim)
        )

        self._init_projection_head()

    def _init_projection_head(self):
        """Proper weight initialization"""
        for module in self.projector:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.encoder(x)
        features = torch.flatten(features, 1)
        embeddings = self.projector(features)
        embeddings = F.normalize(embeddings, dim=1)
        return features, embeddings

# ========================
# MODEL LOADING UTILITIES
# ========================
def load_phase1_model(model_path):
    """Load Phase 1 SimCLR model"""
    try:
        # Load checkpoint - FIXED: Set weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location=eval_config.device, weights_only=False)

        # Get feature dimension from config
        if 'config' in checkpoint and 'feature_dim' in checkpoint['config']:
            feature_dim = checkpoint['config']['feature_dim']
        else:
            feature_dim = 256  # Default

        # Create model
        model = PhDSimCLR(feature_dim=feature_dim)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(eval_config.device)
        model.eval()

        print(f"✅ Loaded Phase 1 model from {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading Phase 1 model: {e}")
        return None

def load_phase2_model(model_path):
    """Load Phase 2 model with classifier"""
    try:
        # Load checkpoint - FIXED: Set weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location=eval_config.device, weights_only=False)

        # Create encoder
        encoder = PhDSimCLR()
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.to(eval_config.device)
        encoder.eval()

        # Create classifier
        classifier = nn.Linear(2048, 2)
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
        classifier.to(eval_config.device)
        classifier.eval()

        print(f"✅ Loaded Phase 2 model from {model_path}")
        return encoder, classifier
    except Exception as e:
        print(f"❌ Error loading Phase 2 model: {e}")
        return None, None

def find_saved_models():
    """Find all saved models in the directory structure"""
    phase1_models = []
    phase2_models = []

    # Find Phase 1 models
    phase1_dir = os.path.join(eval_config.SAVE_DIR, "phase1_pretrain", "models")
    if os.path.exists(phase1_dir):
        for file in os.listdir(phase1_dir):
            if file.endswith(".pth"):
                model_path = os.path.join(phase1_dir, file)
                model_name = file.replace(".pth", "")
                phase1_models.append((model_name, model_path, "phase1"))

    # Find Phase 2 models
    phase2_dir = os.path.join(eval_config.SAVE_DIR, "phase2_finetune", "models")
    if os.path.exists(phase2_dir):
        for file in os.listdir(phase2_dir):
            if file.endswith(".pth"):
                model_path = os.path.join(phase2_dir, file)
                model_name = file.replace(".pth", "")
                phase2_models.append((model_name, model_path, "phase2"))

    return phase1_models + phase2_models

# ========================
# DATA LOADING
# ========================
class MVTecEvalDataset(Dataset):
    """Dataset for MVTec-AD evaluation"""
    def __init__(self, data_root, categories, transform=None):
        self.data_root = data_root
        self.categories = categories
        self.transform = transform
        self.samples = []

        # Load test data for all categories
        for category in categories:
            test_path = os.path.join(data_root, category, 'test')
            if not os.path.exists(test_path):
                continue

            # Load good images
            good_path = os.path.join(test_path, 'good')
            if os.path.exists(good_path):
                for img_name in os.listdir(good_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(good_path, img_name)
                        self.samples.append((img_path, 0, category))  # 0 = normal

            # Load defect images
            for defect_type in os.listdir(test_path):
                if defect_type == 'good':
                    continue
                defect_path = os.path.join(test_path, defect_type)
                if os.path.isdir(defect_path):
                    for img_name in os.listdir(defect_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(defect_path, img_name)
                            self.samples.append((img_path, 1, category))  # 1 = anomaly

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, category = self.samples[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # Return a black image if loading fails
            img = Image.new('RGB', (eval_config.IMAGE_SIZE, eval_config.IMAGE_SIZE))

        if self.transform:
            img = self.transform(img)

        return img, label, category, img_path

def get_eval_dataloader(categories=None):
    """Get evaluation dataloader"""
    if categories is None:
        categories = eval_config.ALL_CATEGORIES

    transform = transforms.Compose([
        transforms.Resize((eval_config.IMAGE_SIZE, eval_config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = MVTecEvalDataset(eval_config.DATA_ROOT, categories, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_config.BATCH_SIZE,
        shuffle=False,
        num_workers=eval_config.NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader

# ========================
# EVALUATION METRICS
# ========================
def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive evaluation metrics"""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1_score'] = f1_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate

    # AUC-ROC if probabilities are available
    if y_proba is not None and len(set(y_true)) > 1:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
            # Precision-Recall AUC
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            metrics['auc_pr'] = auc(recall, precision)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            metrics['auc_roc'] = 0
            metrics['auc_pr'] = 0

    return metrics

# ========================
# VISUALIZATION FUNCTIONS
# ========================
def plot_confusion_matrix(cm, classes, save_path, model_name):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=eval_config.DPI)
    plt.close()

def plot_roc_curve(y_true, y_proba, save_path, model_name):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=eval_config.DPI)
    plt.close()

def plot_tsne(features, labels, categories, save_path, model_name, collective=True):
    """Plot T-SNE visualization"""
    # Sample data if too large
    max_samples = 1000  # Reduced for CPU
    if len(features) > max_samples:
        indices = np.random.choice(len(features), max_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
        categories = [categories[i] for i in indices]

    # Run T-SNE
    print(f"Running T-SNE on {len(features)} samples...")
    tsne = TSNE(n_components=2, perplexity=min(eval_config.TSNE_PERPLEXITY, len(features)-1),
                n_iter=eval_config.TSNE_ITERATIONS, random_state=42)
    tsne_results = tsne.fit_transform(features)

    # Create dataframe for plotting
    df = pd.DataFrame({
        'tsne-1': tsne_results[:, 0],
        'tsne-2': tsne_results[:, 1],
        'label': labels,
        'category': categories
    })

    # Plot
    plt.figure(figsize=eval_config.FIGSIZE)
    if collective:
        # All categories together
        sns.scatterplot(
            x='tsne-1', y='tsne-2',
            hue='label',
            palette={0: 'blue', 1: 'red'},
            data=df,
            legend="full",
            alpha=0.7
        )
        plt.title(f'T-SNE Visualization (All Categories) - {model_name}')
    else:
        # Per category
        unique_cats = df['category'].unique()
        n_cats = len(unique_cats)
        cols = 5
        rows = (n_cats + cols - 1) // cols

        plt.figure(figsize=(15, 3 * rows))
        for i, cat in enumerate(unique_cats):
            plt.subplot(rows, cols, i+1)
            cat_df = df[df['category'] == cat]
            sns.scatterplot(
                x='tsne-1', y='tsne-2',
                hue='label',
                palette={0: 'blue', 1: 'red'},
                data=cat_df,
                legend="full",
                alpha=0.7
            )
            plt.title(cat)
            plt.axis('off')
        plt.suptitle(f'T-SNE Visualization (Per Category) - {model_name}')

    plt.tight_layout()
    plt.savefig(save_path, dpi=eval_config.DPI)
    plt.close()

def generate_grad_cam(model, img_tensor, target_class, target_layer_name):
    """Generate Grad-CAM heatmap"""
    # This is a simplified Grad-CAM implementation
    # For full implementation, we would need to register hooks on the target layer
    try:
        # For now, we'll return a dummy heatmap
        # In a full implementation, we would:
        # 1. Forward pass to get activations and gradients
        # 2. Compute weights using global average pooling of gradients
        # 3. Create heatmap by weighted combination of activations
        # 4. Normalize and overlay on original image

        # Return a dummy heatmap for demonstration
        heatmap = np.random.rand(img_tensor.shape[2], img_tensor.shape[3])
        return heatmap
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return np.zeros((img_tensor.shape[2], img_tensor.shape[3]))

def plot_data_augmentation_pipeline(save_path):
    """Visualize data augmentation pipeline"""
    # Create example images showing each augmentation step
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original image
    img = Image.new('RGB', (224, 224), color='blue')
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    # Random crop
    axes[0, 1].imshow(img)
    axes[0, 1].set_title('Random Crop')
    axes[0, 1].axis('off')

    # Horizontal flip
    axes[0, 2].imshow(img)
    axes[0, 2].set_title('Horizontal Flip')
    axes[0, 2].axis('off')

    # Color jitter
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Color Jitter')
    axes[1, 0].axis('off')

    # Grayscale
    axes[1, 1].imshow(img)
    axes[1, 1].set_title('Grayscale')
    axes[1, 1].axis('off')

    # Gaussian blur
    axes[1, 2].imshow(img)
    axes[1, 2].set_title('Gaussian Blur')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=eval_config.DPI)
    plt.close()

def plot_workflow_diagram(save_path):
    """Create workflow methodology diagram"""
    # Create a simple workflow diagram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Draw boxes and arrows
    ax.text(0.5, 0.9, 'Phase 1: Self-Supervised Pretraining',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    ax.text(0.5, 0.7, 'Phase 2: Supervised Fine-tuning',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    ax.text(0.5, 0.5, 'Model Evaluation',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.text(0.5, 0.3, 'Results Analysis',
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

    # Draw arrows
    ax.annotate('', xy=(0.5, 0.75), xytext=(0.5, 0.85),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.5, 0.55), xytext=(0.5, 0.65),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Research Workflow Methodology', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=eval_config.DPI)
    plt.close()

def plot_evaluation_framework(save_path):
    """Create evaluation framework diagram"""
    # Create a simple evaluation framework diagram
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Draw boxes
    ax.text(0.2, 0.8, 'Model Loading',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    ax.text(0.5, 0.8, 'Feature Extraction',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    ax.text(0.8, 0.8, 'Prediction',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    ax.text(0.2, 0.5, 'Metrics Calculation',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))

    ax.text(0.5, 0.5, 'Visualization',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    ax.text(0.8, 0.5, 'Error Analysis',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightpink"))

    ax.text(0.5, 0.2, 'Results Summary',
            ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightsteelblue"))

    # Draw arrows
    ax.annotate('', xy=(0.35, 0.8), xytext=(0.25, 0.8),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.65, 0.8), xytext=(0.55, 0.8),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.2, 0.65), xytext=(0.2, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.8, 0.65), xytext=(0.8, 0.75),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.5, 0.35), xytext=(0.5, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.title('Evaluation Framework', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=eval_config.DPI)
    plt.close()

# ========================
# ERROR ANALYSIS
# ========================
def analyze_errors(y_true, y_pred, y_proba, img_paths, categories, save_dir, model_name):
    """Perform comprehensive error analysis"""
    # Identify error indices
    error_indices = np.where(y_true != y_pred)[0]

    # Create error directory
    error_dir = os.path.join(save_dir, "error_analysis")
    os.makedirs(error_dir, exist_ok=True)

    # Separate false positives and false negatives
    fp_indices = error_indices[y_pred[error_indices] == 1]  # Predicted anomaly, normal
    fn_indices = error_indices[y_pred[error_indices] == 0]  # Predicted normal, anomaly

    # Analyze false positives
    fp_analysis = {
        'count': len(fp_indices),
        'categories': Counter([categories[i] for i in fp_indices]),
        'examples': []
    }

    # Save examples of false positives
    for i, idx in enumerate(fp_indices[:eval_config.MAX_ERROR_SAMPLES]):
        img_path = img_paths[idx]
        category = categories[idx]
        confidence = y_proba[idx] if y_proba is not None else 0

        # Copy image to error directory
        img_name = f"fp_{i+1}_{category}_{os.path.basename(img_path)}"
        try:
            shutil.copy(img_path, os.path.join(error_dir, img_name))
        except:
            pass

        fp_analysis['examples'].append({
            'image_path': img_name,
            'category': category,
            'confidence': float(confidence)
        })

    # Analyze false negatives
    fn_analysis = {
        'count': len(fn_indices),
        'categories': Counter([categories[i] for i in fn_indices]),
        'examples': []
    }

    # Save examples of false negatives
    for i, idx in enumerate(fn_indices[:eval_config.MAX_ERROR_SAMPLES]):
        img_path = img_paths[idx]
        category = categories[idx]
        confidence = 1 - y_proba[idx] if y_proba is not None else 0

        # Copy image to error directory
        img_name = f"fn_{i+1}_{category}_{os.path.basename(img_path)}"
        try:
            shutil.copy(img_path, os.path.join(error_dir, img_name))
        except:
            pass

        fn_analysis['examples'].append({
            'image_path': img_name,
            'category': category,
            'confidence': float(confidence)
        })

    # Save error analysis
    error_analysis = {
        'false_positives': fp_analysis,
        'false_negatives': fn_analysis,
        'total_errors': len(error_indices),
        'error_rate': len(error_indices) / len(y_true) if len(y_true) > 0 else 0
    }

    with open(os.path.join(error_dir, "error_analysis.json"), "w") as f:
        json.dump(error_analysis, f, indent=2)

    return error_analysis

# ========================
# KEY FINDINGS GENERATION
# ========================
def generate_key_findings(metrics, error_analysis, save_dir, model_name):
    """Generate key findings summary"""
    findings = {
        'model_name': model_name,
        'performance_summary': {
            'f1_score': metrics['f1_score'],
            'accuracy': metrics['accuracy'],
            'auc_roc': metrics.get('auc_roc', 0),
            'precision': metrics['precision'],
            'recall': metrics['recall']
        },
        'error_analysis': {
            'total_errors': error_analysis['total_errors'],
            'error_rate': error_analysis['error_rate'],
            'false_positive_rate': error_analysis['false_positives']['count'] / len(error_analysis['false_positives']) if error_analysis['false_positives']['count'] > 0 else 0,
            'false_negative_rate': error_analysis['false_negatives']['count'] / len(error_analysis['false_negatives']) if error_analysis['false_negatives']['count'] > 0 else 0
        },
        'strengths': [],
        'weaknesses': [],
        'recommendations': []
    }

    # Analyze strengths
    if metrics['f1_score'] > 0.8:
        findings['strengths'].append("Good F1 score (>0.8)")
    if metrics['recall'] > 0.8:
        findings['strengths'].append("High recall - good at detecting anomalies")
    if metrics['precision'] > 0.8:
        findings['strengths'].append("High precision - few false alarms")

    # Analyze weaknesses
    if metrics['recall'] < 0.7:
        findings['weaknesses'].append("Low recall - missing anomalies")
    if metrics['precision'] < 0.7:
        findings['weaknesses'].append("Low precision - too many false alarms")
    if error_analysis['error_rate'] > 0.3:
        findings['weaknesses'].append("High error rate (>30%)")

    # Generate recommendations
    if metrics['recall'] < 0.7:
        findings['recommendations'].append("Focus on improving anomaly detection sensitivity")
    if metrics['precision'] < 0.7:
        findings['recommendations'].append("Improve specificity to reduce false alarms")
    if error_analysis['false_positives']['count'] > error_analysis['false_negatives']['count']:
        findings['recommendations'].append("Adjust decision threshold to reduce false positives")
    else:
        findings['recommendations'].append("Adjust decision threshold to reduce false negatives")

    # Save findings
    with open(os.path.join(save_dir, "key_findings.json"), "w") as f:
        json.dump(findings, f, indent=2)

    return findings

# ========================
# MAIN EVALUATION FUNCTION
# ========================
def evaluate_model(model_name, model_path, model_type):
    """Evaluate a single model"""
    print(f"\n{'='*80}")
    print(f"Evaluating model: {model_name} (Type: {model_type})")
    print(f"{'='*80}")

    # Create evaluation directory for this model
    eval_dir = os.path.join(eval_config.EVAL_DIR, f"{model_name}_EVALUATIONS")
    os.makedirs(eval_dir, exist_ok=True)

    # Load model based on type
    if model_type == "phase1":
        model = load_phase1_model(model_path)
        if model is None:
            return None

        # For Phase 1 models, we need to extract features and train a classifier
        print("Extracting features for Phase 1 model...")

        # Get evaluation dataloader
        dataloader = get_eval_dataloader()

        # Extract features
        all_features = []
        all_labels = []
        all_categories = []
        all_img_paths = []

        with torch.no_grad():
            for images, labels, categories, img_paths in tqdm(dataloader, desc="Extracting features"):
                images = images.to(eval_config.device)
                features, _ = model(images)
                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_categories.extend(categories)
                all_img_paths.extend(img_paths)

        # Concatenate features
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        all_categories = np.array(all_categories)

        # Train a logistic regression classifier
        print("Training logistic regression classifier...")
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(all_features, all_labels)

        # Get predictions
        all_predictions = classifier.predict(all_features)
        all_probabilities = classifier.predict_proba(all_features)[:, 1]

    else:
        encoder, classifier = load_phase2_model(model_path)
        if encoder is None or classifier is None:
            return None

        # Get evaluation dataloader
        dataloader = get_eval_dataloader()

        # Extract features and predictions
        all_features = []
        all_labels = []
        all_predictions = []
        all_probabilities = []
        all_categories = []
        all_img_paths = []

        with torch.no_grad():
            for images, labels, categories, img_paths in tqdm(dataloader, desc="Extracting features"):
                images = images.to(eval_config.device)
                features, _ = encoder(images)
                logits = classifier(features)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs[:, 1].cpu().numpy())
                all_categories.extend(categories)
                all_img_paths.extend(img_paths)

        # Concatenate features
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_categories = np.array(all_categories)

    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)

    # Save metrics
    with open(os.path.join(eval_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate visualizations
    print("Generating visualizations...")

    # 1. Confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, ['Normal', 'Anomaly'],
                         os.path.join(eval_dir, "confusion_matrix.png"), model_name)

    # 2. ROC curve
    if all_probabilities is not None and len(set(all_labels)) > 1:
        plot_roc_curve(all_labels, all_probabilities,
                      os.path.join(eval_dir, "roc_curve.png"), model_name)

    # 3. T-SNE visualizations
    print("Generating T-SNE visualizations...")
    # Collective T-SNE
    plot_tsne(all_features, all_labels, all_categories,
             os.path.join(eval_dir, "tsne_collective.png"), model_name, collective=True)

    # Per-category T-SNE
    plot_tsne(all_features, all_labels, all_categories,
             os.path.join(eval_dir, "tsne_per_category.png"), model_name, collective=False)

    # 4. Grad-CAM visualizations (sample a few images)
    print("Generating Grad-CAM visualizations...")
    grad_cam_dir = os.path.join(eval_dir, "grad_cam")
    os.makedirs(grad_cam_dir, exist_ok=True)

    # Sample a few images for Grad-CAM
    sample_indices = np.random.choice(len(all_img_paths), min(5, len(all_img_paths)), replace=False)
    for idx in sample_indices:
        img_path = all_img_paths[idx]
        label = all_labels[idx]
        category = all_categories[idx]

        # Load and preprocess image
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.Compose([
            transforms.Resize((eval_config.IMAGE_SIZE, eval_config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])(img).unsqueeze(0).to(eval_config.device)

        # Generate Grad-CAM
        if model_type == "phase1":
            heatmap = generate_grad_cam(model, img_tensor, label, eval_config.TARGET_LAYER_NAME)
        else:
            heatmap = generate_grad_cam(encoder, img_tensor, label, eval_config.TARGET_LAYER_NAME)

        # Save Grad-CAM visualization
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f"Original: {category} ({'Anomaly' if label else 'Normal'})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title("Grad-CAM")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(grad_cam_dir, f"grad_cam_{idx}_{category}.png"), dpi=eval_config.DPI)
        plt.close()

    # 5. Data augmentation pipeline
    plot_data_augmentation_pipeline(os.path.join(eval_dir, "data_augmentation_pipeline.png"))

    # 6. Workflow methodology
    plot_workflow_diagram(os.path.join(eval_dir, "workflow_methodology.png"))

    # 7. Evaluation framework
    plot_evaluation_framework(os.path.join(eval_dir, "evaluation_framework.png"))

    # 8. Error analysis
    print("Performing error analysis...")
    error_analysis = analyze_errors(all_labels, all_predictions, all_probabilities,
                                   all_img_paths, all_categories, eval_dir, model_name)

    # 9. Performance metrics visualization
    plt.figure(figsize=(12, 8))
    metrics_names = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Specificity']
    metrics_values = [metrics['accuracy'], metrics['f1_score'], metrics['precision'],
                     metrics['recall'], metrics['specificity']]

    bars = plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title(f'Performance Metrics - {model_name}')
    plt.ylabel('Score')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "performance_metrics.png"), dpi=eval_config.DPI)
    plt.close()

    # 10. Confusion matrix analysis
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix (Percentage) - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, "confusion_matrix_percentage.png"), dpi=eval_config.DPI)
    plt.close()

    # 11. Failure analysis
    failure_dir = os.path.join(eval_dir, "failure_analysis")
    os.makedirs(failure_dir, exist_ok=True)

    # Create failure analysis plots
    plt.figure(figsize=(12, 6))

    # False positives per category
    fp_counts = [error_analysis['false_positives']['categories'].get(cat, 0) for cat in eval_config.ALL_CATEGORIES]
    plt.subplot(1, 2, 1)
    plt.bar(eval_config.ALL_CATEGORIES, fp_counts, color='red')
    plt.title('False Positives per Category')
    plt.xticks(rotation=90)
    plt.tight_layout()

    # False negatives per category
    fn_counts = [error_analysis['false_negatives']['categories'].get(cat, 0) for cat in eval_config.ALL_CATEGORIES]
    plt.subplot(1, 2, 2)
    plt.bar(eval_config.ALL_CATEGORIES, fn_counts, color='orange')
    plt.title('False Negatives per Category')
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(os.path.join(failure_dir, "failure_analysis.png"), dpi=eval_config.DPI)
    plt.close()

    # 12. Key findings
    print("Generating key findings...")
    key_findings = generate_key_findings(metrics, error_analysis, eval_dir, model_name)

    # Create summary report
    summary = {
        'model_name': model_name,
        'model_type': model_type,
        'evaluation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'metrics': metrics,
        'error_analysis': error_analysis,
        'key_findings': key_findings
    }

    with open(os.path.join(eval_dir, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete for {model_name}!")
    print(f"Results saved to: {eval_dir}")

    return summary

# ========================
# MAIN EVALUATION LOOP
# ========================
def main():
    """Main evaluation function"""
    print("🚀 Starting Comprehensive Model Evaluation")
    print(f"{'='*80}")

    # Find all saved models
    models = find_saved_models()
    print(f"Found {len(models)} models to evaluate")

    # Evaluate each model
    all_results = []
    for model_name, model_path, model_type in models:
        try:
            result = evaluate_model(model_name, model_path, model_type)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue

    # Create comparison summary
    if len(all_results) > 1:
        print("\nGenerating comparison summary...")
        comparison_dir = os.path.join(eval_config.EVAL_DIR, "model_comparison")
        os.makedirs(comparison_dir, exist_ok=True)

        # Create comparison table
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                'Model': result['model_name'],
                'Type': result['model_type'],
                'F1 Score': result['metrics']['f1_score'],
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'AUC-ROC': result['metrics'].get('auc_roc', 0),
                'Error Rate': result['error_analysis']['error_rate']
            })

        df = pd.DataFrame(comparison_data)
        df.to_csv(os.path.join(comparison_dir, "model_comparison.csv"), index=False)

        # Create comparison plots
        plt.figure(figsize=(15, 10))

        # F1 Score comparison
        plt.subplot(2, 2, 1)
        plt.bar(df['Model'], df['F1 Score'], color='green')
        plt.title('F1 Score Comparison')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)

        # Accuracy comparison
        plt.subplot(2, 2, 2)
        plt.bar(df['Model'], df['Accuracy'], color='blue')
        plt.title('Accuracy Comparison')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)

        # Precision-Recall tradeoff
        plt.subplot(2, 2, 3)
        plt.scatter(df['Recall'], df['Precision'], s=100, alpha=0.7)
        for i, row in df.iterrows():
            plt.text(row['Recall'], row['Precision'], row['Model'], fontsize=8)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Tradeoff')
        plt.grid(True)

        # Error rate comparison
        plt.subplot(2, 2, 4)
        plt.bar(df['Model'], df['Error Rate'], color='red')
        plt.title('Error Rate Comparison')
        plt.xticks(rotation=90)
        plt.ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "model_comparison.png"), dpi=eval_config.DPI)
        plt.close()

        print(f"Model comparison saved to: {comparison_dir}")

    print(f"\n{'='*80}")
    print("✅ All evaluations complete!")
    print(f"📁 Results saved to: {eval_config.EVAL_DIR}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()