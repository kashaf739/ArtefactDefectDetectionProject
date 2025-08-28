from google.colab import drive
drive.mount('/content/drive')

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings("ignore")

# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================
DATASET_ROOT = "/content/drive/MyDrive/Colab/ExtractedDataset/mvtec_ad_unzipped/mvtec_ad"
OUTPUT_DIR = "/content/drive/MyDrive/Nubia2025"

class Config:
    """Configuration class for hyperparameters and settings."""
    # Paths
    DATASET_ROOT = DATASET_ROOT
    OUTPUT_DIR = OUTPUT_DIR

    # Model & Training
    MODEL_NAME = "resnet50"
    NUM_EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4

    # Data
    IMG_SIZE = 224
    VALIDATION_SPLIT = 0.2
    RANDOM_STATE = 42

    # Labels
    LABEL_MAP = {"good": 0, "defective": 1}
    INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    # Training
    PATIENCE = 5  # For early stopping
    MAX_LR = 0.01  # For OneCycleLR scheduler

# Create output directory if it doesn't exist
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"Configuration loaded. Outputs will be saved to: {Config.OUTPUT_DIR}")

# =====================================================================================
# 2. DATA PREPARATION (DYNAMIC PATH SCANNING)
# =====================================================================================
def prepare_data_paths(root_dir):
    """
    Scans the MVTec AD directory and creates a list of image paths and corresponding binary labels.
    """
    print("Scanning dataset directory to prepare file paths and labels...")
    image_paths = []
    labels = []

    try:
        category_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    except Exception as e:
        raise RuntimeError(f"Error accessing dataset directory: {str(e)}. "
                          f"Check if path '{root_dir}' exists and contains data.") from e

    for category in tqdm(category_dirs, desc="Processing Categories"):
        category_path = os.path.join(root_dir, category)

        # --- Process 'good' images ---
        train_good_path = os.path.join(category_path, "train", "good")
        test_good_path = os.path.join(category_path, "test", "good")

        for good_path in [train_good_path, test_good_path]:
            if os.path.exists(good_path):
                for img_file in os.listdir(good_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_paths.append(os.path.join(good_path, img_file))
                        labels.append(Config.LABEL_MAP["good"])

        # --- Process 'defective' images ---
        test_dir = os.path.join(category_path, "test")
        if os.path.exists(test_dir):
            defect_types = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d)) and d != "good"]
            for defect in defect_types:
                defect_path = os.path.join(test_dir, defect)
                for img_file in os.listdir(defect_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_paths.append(os.path.join(defect_path, img_file))
                        labels.append(Config.LABEL_MAP["defective"])

    print(f"Scan complete. Found {len(image_paths)} images.")
    print(f" -> Good images: {labels.count(0)}")
    print(f" -> Defective images: {labels.count(1)}")
    return image_paths, labels

# =====================================================================================
# 3. CUSTOM PYTORCH DATASET WITH ALBUMENTATIONS
# =====================================================================================
class MVTecDataset(Dataset):
    """Custom Dataset for loading MVTec images with Albumentations transformations."""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            label = self.labels[idx]

            if self.transform:
                # Convert PIL to NumPy array for Albumentations
                image_np = np.array(image)
                transformed = self.transform(image=image_np)
                image_tensor = transformed['image']
            else:
                image_tensor = transforms.ToTensor()(image)

            return image_tensor, label

        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return dummy data instead of failing
            return torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE)), 0

# =====================================================================================
# 4. ENHANCED MODEL ARCHITECTURE
# =====================================================================================
def get_model():
    """Loads a pre-trained ResNet-50 with enhanced architecture."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Unfreeze last two residual blocks for fine-tuning
    for name, child in model.named_children():
        if name in ['layer3', 'layer4']:
            for param in child.parameters():
                param.requires_grad = True
        else:
            for param in child.parameters():
                param.requires_grad = False

    # Replace final fully connected layer with dropout and additional layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, len(Config.LABEL_MAP))
    )

    return model

# =====================================================================================
# 5. HELPER FUNCTIONS FOR EVALUATION AND SAVING
# =====================================================================================
def save_confusion_matrix(y_true, y_pred, epoch, output_dir):
    """Calculates, plots, and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.LABEL_MAP.keys(),
                yticklabels=Config.LABEL_MAP.keys())
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - Epoch {epoch}')

    save_path = os.path.join(output_dir, f"confusion_matrix_epoch_{epoch}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix for epoch {epoch} saved to {save_path}")

def save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, val_f1, output_dir):
    """Save training metrics to a text file."""
    metrics_path = os.path.join(output_dir, "training_metrics.txt")
    with open(metrics_path, 'a') as f:
        f.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}\n")

# =====================================================================================
# 6. MAIN TRAINING SCRIPT WITH IMPROVEMENTS
# =====================================================================================
def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Splitting ---
    try:
        all_paths, all_labels = prepare_data_paths(Config.DATASET_ROOT)
    except RuntimeError as e:
        print(f"Data preparation failed: {str(e)}")
        return

    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=Config.VALIDATION_SPLIT,
        random_state=Config.RANDOM_STATE,
        stratify=all_labels
    )

    # --- Advanced Data Augmentation ---
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # --- Create Datasets and DataLoaders ---
    train_dataset = MVTecDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = MVTecDataset(val_paths, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Training data: {len(train_dataset)} samples")
    print(f"Validation data: {len(val_dataset)} samples")

    # --- Model, Loss, and Optimizer ---
    model = get_model().to(device)

    # Calculate class weights for imbalanced dataset
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = torch.tensor([1.0 / class_counts[0], 1.0 / class_counts[1]], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AdamW optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                           weight_decay=Config.WEIGHT_DECAY)

    # OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=Config.MAX_LR,
        steps_per_epoch=len(train_loader),
        epochs=Config.NUM_EPOCHS
    )

    # Early stopping variables
    best_f1 = 0.0
    patience_counter = 0

    # --- Training Loop ---
    start_time = time.time()
    for epoch in range(1, Config.NUM_EPOCHS + 1):
        print("-" * 50)
        print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")

        # -- Training Phase --
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()  # Update learning rate

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # -- Validation Phase --
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average='binary')

        print(f'Validation Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f} F1: {val_f1:.4f}')

        # --- Save Checkpoint and Metrics ---
        model_save_path = os.path.join(Config.OUTPUT_DIR, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Model checkpoint saved to {model_save_path}")

        save_confusion_matrix(all_labels, all_preds, epoch, Config.OUTPUT_DIR)
        save_metrics(epoch, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, val_f1, Config.OUTPUT_DIR)

        # --- Early Stopping Check ---
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            # Save best model separately
            best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs")
                break

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation F1-score: {best_f1:.4f}")

if __name__ == '__main__':
    # Check if dataset exists
    if not os.path.exists(Config.DATASET_ROOT) or len(os.listdir(Config.DATASET_ROOT)) == 0:
        print("\n" + "="*60)
        print("!!! CRITICAL ERROR: DATASET DIRECTORY IS EMPTY OR DOESN'T EXIST !!!")
        print(f"Path: {Config.DATASET_ROOT}")
        print("Please upload the MVTec AD dataset to this location.")
        print("="*60 + "\n")
    else:
        try:
            main()
        except Exception as e:
            print(f"\nScript execution failed with error: {str(e)}")
            print("Please check:")
            print("1. GPU availability (Runtime → Change runtime type)")
            print("2. Dataset integrity")
            print("3. Folder permissions")