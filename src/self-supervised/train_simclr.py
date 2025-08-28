import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import gc
from datetime import datetime
import json
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
import seaborn as sns
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
warnings.filterwarnings('ignore')

# Disable mixed precision to avoid fp16 overflow issues
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ======================== GLOBAL CONFIGURATION ========================
class Config:
    """PhD Research Configuration for SimCLR on MVTec AD"""

    # ALL 15 MVTEC-AD CATEGORIES
    ALL_CATEGORIES = [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper"
    ]

    # PATHS - ADJUST THESE FOR YOUR GOOGLE COLAB SETUP
    DATA_ROOT = "/content/drive/MyDrive/MVTEC_AD/mvtec_ad"
    SAVE_DIR = "/content/drive/MyDrive/PhD_SimCLR_Results"
    MODEL_NAME = "PhD_SimCLR_MVTecAD"

    # PHASE 1: SELF-SUPERVISED PRETRAINING PARAMETERS
    PRETRAIN_EPOCHS = 7
    PRETRAIN_BATCH_SIZE = 48  # Reduced for T4 stability
    IMAGE_SIZE = 224
    BASE_LR = 1e-4  # More conservative LR
    MIN_LR = 1e-7
    WEIGHT_DECAY = 1e-4
    FEATURE_DIM = 256
    TEMPERATURE = 0.07
    WARMUP_EPOCHS = 5  # Reduced warmup
    DATA_MULTIPLIER = 4  # Reduced multiplier

    # PHASE 2: FINE-TUNING PARAMETERS
    FINETUNE_EPOCHS = 30  # Reduced epochs
    FINETUNE_BATCH_SIZE = 32
    FINETUNE_LR = 1e-4
    LABELED_DATA_RATIO = 0.1  # 10% labeled data as requested

    # PhD RESEARCH TARGETS
    TARGET_F1_SCORE = 0.90  # F1 > 90% as specified

    # TRAINING OPTIMIZATIONS
    GRAD_CLIP = 1.0
    VALIDATION_INTERVAL = 5  # More frequent validation
    SAVE_EVERY_EPOCH = True

    def __init__(self):
        self.device = self._setup_device()
        self.active_categories = []

    def _setup_device(self):
        """Setup T4 GPU with conservative settings"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

            # Conservative memory usage for T4
            torch.cuda.set_per_process_memory_fraction(0.8)

            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9

            print(f"🚀 GPU: {gpu_name}")
            print(f"💾 Memory: {gpu_memory:.1f} GB")

            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()

        else:
            device = torch.device("cpu")
            print("⚠️ CUDA not available! Using CPU")
            # Reduce parameters for CPU
            self.PRETRAIN_BATCH_SIZE = 16
            self.FINETUNE_BATCH_SIZE = 16
            self.DATA_MULTIPLIER = 2

        return device

# Seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Initialize configuration
config = Config()
seed_everything(42)

print("🎓 PhD-LEVEL SIMCLR FOR MVTEC-AD ANOMALY DETECTION")
print("=" * 80)
print(f"🎯 Research Target: F1-Score > {config.TARGET_F1_SCORE:.0%}")
print(f"📊 Dataset: MVTec-AD (15 categories)")
print(f"🔬 Method: Two-Phase SimCLR (Self-Supervised → Fine-Tuned)")
print(f"💡 Innovation: {config.LABELED_DATA_RATIO:.0%} labeled data for PhD-level efficiency")
print(f"🚀 Hardware: {config.device}")
print("=" * 80)

# ======================== DIRECTORY SETUP ========================
def create_directory_structure():
    """Create comprehensive directory structure for PhD research"""
    try:
        dirs_to_create = [
            config.SAVE_DIR,
            os.path.join(config.SAVE_DIR, "phase1_pretrain", "models"),
            os.path.join(config.SAVE_DIR, "phase1_pretrain", "metrics"),
            os.path.join(config.SAVE_DIR, "phase1_pretrain", "plots"),
            os.path.join(config.SAVE_DIR, "phase2_finetune", "models"),
            os.path.join(config.SAVE_DIR, "phase2_finetune", "metrics"),
            os.path.join(config.SAVE_DIR, "phase2_finetune", "plots"),
            os.path.join(config.SAVE_DIR, "evaluation", "results"),
            os.path.join(config.SAVE_DIR, "evaluation", "visualizations"),
            os.path.join(config.SAVE_DIR, "evaluation", "per_category")
        ]

        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)

        # Save research configuration
        research_config = {
            "research_title": "SimCLR for Industrial Anomaly Detection",
            "dataset": "MVTec-AD",
            "target_f1": config.TARGET_F1_SCORE,
            "phase1_epochs": config.PRETRAIN_EPOCHS,
            "phase2_epochs": config.FINETUNE_EPOCHS,
            "labeled_data_ratio": config.LABELED_DATA_RATIO,
            "feature_dim": config.FEATURE_DIM,
            "temperature": config.TEMPERATURE,
            "device": str(config.device),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(os.path.join(config.SAVE_DIR, "research_config.json"), "w") as f:
            json.dump(research_config, f, indent=2)

        print(f"📁 Research directories created in: {config.SAVE_DIR}")
        return True

    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return False

# ======================== DATA DISCOVERY & VALIDATION ========================
def discover_mvtec_data():
    """Comprehensive MVTec-AD data discovery and validation"""
    print(f"\n🔍 Discovering MVTec-AD dataset...")

    train_data = {}
    test_data = {}
    category_stats = {}

    for i, category in enumerate(config.ALL_CATEGORIES):
        try:
            # Phase 1: Training data (good images only)
            train_good_path = os.path.join(config.DATA_ROOT, category, 'train', 'good')
            train_good_images = []

            if os.path.exists(train_good_path):
                for filename in os.listdir(train_good_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        full_path = os.path.join(train_good_path, filename)
                        try:
                            if os.path.getsize(full_path) > 1000:
                                with Image.open(full_path) as img:
                                    if img.size[0] > 50 and img.size[1] > 50:
                                        train_good_images.append(full_path)
                        except Exception:
                            continue

            # Phase 2: Test data (good + defective)
            test_path = os.path.join(config.DATA_ROOT, category, 'test')
            test_good_images = []
            test_defect_images = []

            if os.path.exists(test_path):
                # Test good images
                test_good_path = os.path.join(test_path, 'good')
                if os.path.exists(test_good_path):
                    for filename in os.listdir(test_good_path):
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                            full_path = os.path.join(test_good_path, filename)
                            try:
                                if os.path.getsize(full_path) > 1000:
                                    with Image.open(full_path) as img:
                                        if img.size[0] > 50 and img.size[1] > 50:
                                            test_good_images.append((full_path, 0))
                            except Exception:
                                continue

                # Test defect images
                for defect_type in os.listdir(test_path):
                    defect_path = os.path.join(test_path, defect_type)
                    if defect_type != 'good' and os.path.isdir(defect_path):
                        for filename in os.listdir(defect_path):
                            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                full_path = os.path.join(defect_path, filename)
                                try:
                                    if os.path.getsize(full_path) > 1000:
                                        with Image.open(full_path) as img:
                                            if img.size[0] > 50 and img.size[1] > 50:
                                                test_defect_images.append((full_path, 1))
                                except Exception:
                                    continue

            # Store data if category has sufficient images
            if len(train_good_images) > 10:
                train_data[category] = train_good_images
                test_data[category] = test_good_images + test_defect_images
                config.active_categories.append(category)

                category_stats[category] = {
                    'train_good': len(train_good_images),
                    'test_good': len(test_good_images),
                    'test_defect': len(test_defect_images),
                    'total_test': len(test_good_images) + len(test_defect_images)
                }

                print(f"✅ [{i+1:2d}/15] {category:<12}: Train={len(train_good_images):>3d}, Test={len(test_good_images)+len(test_defect_images):>3d} ({len(test_defect_images)} defects)")
            else:
                print(f"❌ [{i+1:2d}/15] {category:<12}: Insufficient data")

        except Exception as e:
            print(f"❌ [{i+1:2d}/15] {category:<12}: Error - {str(e)[:40]}...")

    # Summary
    total_train = sum(len(imgs) for imgs in train_data.values())
    total_test = sum(len(imgs) for imgs in test_data.values())

    print(f"\n📊 DATASET SUMMARY:")
    print(f"   ✅ Active categories: {len(config.active_categories)}/15")
    print(f"   🔹 Phase 1 (train/good): {total_train:,} images")
    print(f"   🔹 Phase 2 (test): {total_test:,} images")

    if len(config.active_categories) == 0:
        raise ValueError("❌ CRITICAL: No valid categories found! Check DATA_ROOT path.")

    return train_data, test_data, category_stats

# ======================== OPTIMIZED AUGMENTATIONS ========================
class SimCLRAugmentation:
    """PhD-optimized augmentations for SimCLR"""

    def __init__(self, image_size=224, strength=1.0):
        self.size = image_size

        # Fixed kernel size for GaussianBlur
        blur_kernel = 23  # Fixed odd kernel size

        self.transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4*strength, 0.4*strength, 0.4*strength, 0.1*strength)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transform(img)

class SimpleAugmentation:
    """Simple augmentation for fine-tuning phase"""

    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transform(img)

# ======================== DATASETS ========================
class Phase1Dataset(Dataset):
    """Phase 1: Self-supervised pretraining on good images only"""

    def __init__(self, image_paths, augmentation, multiplier=4):
        self.paths = image_paths
        self.aug = augmentation
        self.multiplier = multiplier

        print(f"📊 Phase 1 Dataset: {len(image_paths):,} images × {multiplier} = {len(image_paths) * multiplier:,} samples")

    def __len__(self):
        return len(self.paths) * self.multiplier

    def __getitem__(self, idx):
        actual_idx = idx % len(self.paths)
        path = self.paths[actual_idx]

        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            backup_idx = random.randint(0, len(self.paths) - 1)
            img = Image.open(self.paths[backup_idx]).convert('RGB')

        # Generate two augmented views for contrastive learning
        view1 = self.aug(img)
        view2 = self.aug(img)

        return view1, view2

class Phase2Dataset(Dataset):
    """Phase 2: Supervised fine-tuning with labeled test data"""

    def __init__(self, image_paths_with_labels, augmentation):
        self.data = image_paths_with_labels
        self.aug = augmentation

        normal_count = sum(1 for _, label in self.data if label == 0)
        anomaly_count = sum(1 for _, label in self.data if label == 1)

        print(f"📊 Phase 2 Dataset: {len(self.data):,} images (Normal: {normal_count}, Anomaly: {anomaly_count})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]

        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            backup_idx = random.randint(0, len(self.data) - 1)
            path, label = self.data[backup_idx]
            img = Image.open(path).convert('RGB')

        img_tensor = self.aug(img)
        return img_tensor, torch.tensor(label, dtype=torch.long)

# ======================== SIMCLR MODEL ========================
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

# ======================== FIXED NT-XENT LOSS ========================
class StableNTXentLoss(nn.Module):
    """Fixed NT-Xent loss without fp16 overflow issues"""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Ensure L2 normalization and fp32 computation
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)

        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix in fp32
        sim_matrix = torch.matmul(z, z.T) / self.temperature

        # Create mask to remove self-similarity
        mask = torch.eye(2*N, device=z.device, dtype=torch.bool)

        # Use a smaller negative value that works with fp16
        sim_matrix = sim_matrix.masked_fill(mask, -1e4)

        # Positive pairs indices
        targets = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)

        # Stable cross-entropy
        loss = F.cross_entropy(sim_matrix, targets)
        return loss

# ======================== METRICS TRACKING ========================
class PhDMetricsTracker:
    """Comprehensive metrics tracking"""

    def __init__(self, phase_name):
        self.phase_name = phase_name
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.f1_scores = []
        self.auroc_scores = []
        self.best_f1 = 0.0
        self.best_epoch = 0

    def update(self, epoch, train_loss, val_loss=None, lr=None, f1=None, auroc=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if f1 is not None:
            self.f1_scores.append(f1)
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_epoch = epoch
        if auroc is not None:
            self.auroc_scores.append(auroc)

    def save_plots(self, save_dir):
        """Save training plots"""
        try:
            plt.figure(figsize=(15, 10))

            # Loss plot
            plt.subplot(2, 3, 1)
            plt.plot(self.train_losses, label='Train Loss', linewidth=2)
            if self.val_losses:
                val_epochs = list(range(config.VALIDATION_INTERVAL-1,
                                      len(self.val_losses)*config.VALIDATION_INTERVAL,
                                      config.VALIDATION_INTERVAL))
                plt.plot(val_epochs, self.val_losses, label='Val Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title(f'{self.phase_name} - Training Loss')
            plt.grid(True, alpha=0.3)

            # Learning rate
            plt.subplot(2, 3, 2)
            if self.learning_rates:
                plt.plot(self.learning_rates, linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title(f'{self.phase_name} - Learning Rate Schedule')
            plt.grid(True, alpha=0.3)
            plt.yscale('log')

            # F1 Score evolution
            if self.f1_scores:
                plt.subplot(2, 3, 3)
                plt.plot(self.f1_scores, linewidth=2, color='green')
                plt.axhline(y=config.TARGET_F1_SCORE, color='red', linestyle='--',
                           label=f'Target ({config.TARGET_F1_SCORE:.0%})')
                plt.xlabel('Evaluation')
                plt.ylabel('F1 Score')
                plt.title(f'{self.phase_name} - F1 Score Progress')
                plt.legend()
                plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{self.phase_name.lower()}_metrics.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"⚠️ Error saving plots: {e}")

# ======================== PHASE 1: SELF-SUPERVISED PRETRAINING ========================
def phase1_pretrain(train_data):
    """Phase 1: Self-supervised pretraining"""

    print("\n" + "="*60)
    print("🔬 PHASE 1: SELF-SUPERVISED PRETRAINING")
    print("="*60)

    # Flatten all training paths
    all_train_paths = []
    for category, paths in train_data.items():
        all_train_paths.extend(paths)

    print(f"📊 Training on {len(all_train_paths):,} good images from {len(train_data)} categories")

    # Create augmentations and dataset
    train_aug = SimCLRAugmentation(config.IMAGE_SIZE, strength=1.0)
    val_aug = SimCLRAugmentation(config.IMAGE_SIZE, strength=0.3)

    # Train/validation split
    random.shuffle(all_train_paths)
    split_idx = int(0.85 * len(all_train_paths))
    train_paths = all_train_paths[:split_idx]
    val_paths = all_train_paths[split_idx:]

    # Create datasets
    train_dataset = Phase1Dataset(train_paths, train_aug, config.DATA_MULTIPLIER)
    val_dataset = Phase1Dataset(val_paths, val_aug, multiplier=2)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )

    # Initialize model
    model = PhDSimCLR(config.FEATURE_DIM).to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"🧠 Model Architecture:")
    print(f"   Backbone: ResNet50")
    print(f"   Feature Dimension: {config.FEATURE_DIM}")
    print(f"   Total Parameters: {total_params:,}")
    print(f"   Trainable Parameters: {trainable_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.BASE_LR,
        weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.PRETRAIN_EPOCHS, eta_min=config.MIN_LR
    )

    # Loss function
    criterion = StableNTXentLoss(config.TEMPERATURE)

    # Metrics tracker
    metrics = PhDMetricsTracker("Phase1_Pretraining")

    print(f"\n🎯 Starting pretraining for {config.PRETRAIN_EPOCHS} epochs...")

    phase1_pretrain.best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, config.PRETRAIN_EPOCHS + 1):

        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Phase 1 - Epoch {epoch:03d}/{config.PRETRAIN_EPOCHS}")

        for batch_idx, (x1, x2) in enumerate(progress_bar):
            x1, x2 = x1.to(config.device, non_blocking=True), x2.to(config.device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass (no mixed precision to avoid fp16 issues)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/num_batches:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })

            # Memory cleanup
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = total_loss / num_batches
        current_lr = scheduler.get_last_lr()[0]

        # Validation phase
        val_loss = None
        if epoch % config.VALIDATION_INTERVAL == 0 or epoch == config.PRETRAIN_EPOCHS:
            model.eval()
            total_val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for val_x1, val_x2 in val_loader:
                    val_x1, val_x2 = val_x1.to(config.device), val_x2.to(config.device)

                    _, val_z1 = model(val_x1)
                    _, val_z2 = model(val_x2)
                    val_batch_loss = criterion(val_z1, val_z2)

                    total_val_loss += val_batch_loss.item()
                    val_batches += 1

                    if val_batches >= 20:  # Limit validation batches
                        break

            val_loss = total_val_loss / val_batches

        # Update scheduler
        scheduler.step()

        # Update metrics
        metrics.update(epoch, avg_train_loss, val_loss, current_lr)

        # Save model
        if config.SAVE_EVERY_EPOCH:
            try:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'config': {
                        'feature_dim': config.FEATURE_DIM,
                        'temperature': config.TEMPERATURE,
                        'categories': config.active_categories
                    }
                }

                # Save epoch checkpoint
                epoch_path = os.path.join(config.SAVE_DIR, "phase1_pretrain", "models",
                                        f"phase1_epoch_{epoch:03d}.pth")
                torch.save(checkpoint, epoch_path)

                # Save best model
                if val_loss is not None and val_loss < phase1_pretrain.best_val_loss:
                    phase1_pretrain.best_val_loss = val_loss
                    best_path = os.path.join(config.SAVE_DIR, "phase1_pretrain", "models",
                                           "phase1_best.pth")
                    torch.save(checkpoint, best_path)
                    print(f"🏆 NEW BEST PHASE 1 MODEL! Val Loss: {val_loss:.4f}")

            except Exception as e:
                print(f"⚠️ Error saving Phase 1 model: {e}")

        # Display progress
        print(f"\n📊 Phase 1 - Epoch {epoch} Results:")
        print(f"   📈 Train Loss: {avg_train_loss:.6f}")
        if val_loss is not None:
            print(f"   📉 Val Loss: {val_loss:.6f}")
        print(f"   🎯 Learning Rate: {current_lr:.2e}")

        # Save metrics periodically
        if epoch % 10 == 0 or epoch == config.PRETRAIN_EPOCHS:
            try:
                plots_dir = os.path.join(config.SAVE_DIR, "phase1_pretrain", "plots")
                metrics.save_plots(plots_dir)
            except Exception as e:
                print(f"⚠️ Error saving Phase 1 metrics: {e}")

        # Memory cleanup
        gc.collect()
        torch.cuda.empty_cache()

    # Phase 1 complete
    print(f"\n🎉 PHASE 1 COMPLETE!")
    print(f"   🏆 Best Validation Loss: {phase1_pretrain.best_val_loss:.6f}")
    print(f"   💾 Models saved in: phase1_pretrain/models/")

    return model, metrics

# ======================== PHASE 2: SUPERVISED FINE-TUNING & EVALUATION ========================
def phase2_finetune_and_evaluate(pretrained_model, test_data):
    """Phase 2: Fine-tune with labeled data and evaluate"""

    print("\n" + "="*60)
    print("🎯 PHASE 2: SUPERVISED FINE-TUNING & EVALUATION")
    print("="*60)

    # Freeze encoder and create classifier
    pretrained_model.eval()
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # Create linear classifier for anomaly detection
    classifier = nn.Linear(2048, 2).to(config.device)

    # Prepare test data with 10% labeling
    all_test_data = []

    for category, test_images in test_data.items():
        if len(test_images) > 0:
            # Use only 10% of labeled data as requested
            sample_size = max(1, int(len(test_images) * config.LABELED_DATA_RATIO))
            sampled_data = random.sample(test_images, sample_size)
            all_test_data.extend(sampled_data)

    print(f"📊 Using {config.LABELED_DATA_RATIO:.0%} labeled data: {len(all_test_data):,} samples")

    # Split into train/test for fine-tuning
    random.shuffle(all_test_data)
    split_idx = int(0.7 * len(all_test_data))
    finetune_data = all_test_data[:split_idx]
    evaluation_data = all_test_data[split_idx:]

    # Create datasets
    simple_aug = SimpleAugmentation(config.IMAGE_SIZE)
    finetune_dataset = Phase2Dataset(finetune_data, simple_aug)
    eval_dataset = Phase2Dataset(evaluation_data, simple_aug)

    # Data loaders
    finetune_loader = DataLoader(finetune_dataset, batch_size=config.FINETUNE_BATCH_SIZE,
                                shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=config.FINETUNE_BATCH_SIZE,
                            shuffle=False, num_workers=2)

    # Optimizer for classifier only
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.FINETUNE_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.FINETUNE_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # Metrics tracker
    metrics = PhDMetricsTracker("Phase2_Finetune")

    print(f"🎯 Fine-tuning classifier for {config.FINETUNE_EPOCHS} epochs...")

    # Fine-tuning loop
    for epoch in range(1, config.FINETUNE_EPOCHS + 1):

        # Training phase
        classifier.train()
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0

        progress_bar = tqdm(finetune_loader, desc=f"Phase 2 - Epoch {epoch:03d}/{config.FINETUNE_EPOCHS}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(config.device), labels.to(config.device)

            optimizer.zero_grad()

            # Extract features with frozen encoder
            with torch.no_grad():
                features, _ = pretrained_model(images)

            # Classify
            logits = classifier(features)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)

            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct_preds/total_preds:.1f}%'
            })

        avg_train_loss = total_loss / len(finetune_loader)
        train_accuracy = 100. * correct_preds / total_preds

        # Evaluation phase
        f1, auroc, precision, recall = evaluate_model(pretrained_model, classifier, eval_loader)

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Update metrics
        metrics.update(epoch, avg_train_loss, None, current_lr, f1, auroc)

        # Save model
        if config.SAVE_EVERY_EPOCH:
            try:
                checkpoint = {
                    'epoch': epoch,
                    'encoder_state_dict': pretrained_model.state_dict(),
                    'classifier_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1_score': f1,
                    'auroc': auroc,
                    'train_loss': avg_train_loss
                }

                epoch_path = os.path.join(config.SAVE_DIR, "phase2_finetune", "models",
                                        f"phase2_epoch_{epoch:03d}.pth")
                torch.save(checkpoint, epoch_path)

                # Save best model based on F1 score
                if f1 > metrics.best_f1:
                    best_path = os.path.join(config.SAVE_DIR, "phase2_finetune", "models",
                                           "phase2_best.pth")
                    torch.save(checkpoint, best_path)
                    print(f"🏆 NEW BEST F1 SCORE: {f1:.4f}")

            except Exception as e:
                print(f"⚠️ Error saving Phase 2 model: {e}")

        # Display results
        print(f"\n📊 Phase 2 - Epoch {epoch} Results:")
        print(f"   📈 Train Loss: {avg_train_loss:.4f}")
        print(f"   🎯 Train Accuracy: {train_accuracy:.2f}%")
        print(f"   🏆 F1 Score: {f1:.4f}")
        print(f"   📊 AUROC: {auroc:.4f}")
        print(f"   ✅ Precision: {precision:.4f}")
        print(f"   🔍 Recall: {recall:.4f}")

        # Check if target achieved
        if f1 > config.TARGET_F1_SCORE:
            print(f"🎊 TARGET ACHIEVED! F1 Score ({f1:.4f}) > {config.TARGET_F1_SCORE:.4f}")

    # Final comprehensive evaluation
    print(f"\n🔬 COMPREHENSIVE EVALUATION...")
    final_results = comprehensive_evaluation(pretrained_model, classifier, test_data)

    # Save Phase 2 results
    try:
        plots_dir = os.path.join(config.SAVE_DIR, "phase2_finetune", "plots")
        metrics.save_plots(plots_dir)
    except Exception as e:
        print(f"⚠️ Error saving Phase 2 plots: {e}")

    return classifier, metrics, final_results

# ======================== EVALUATION FUNCTIONS ========================
def evaluate_model(encoder, classifier, eval_loader):
    """Evaluate model performance"""
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(config.device), labels.to(config.device)

            # Extract features
            features, _ = encoder(images)

            # Classify
            logits = classifier(features)
            probs = F.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of anomaly class

    # Calculate metrics
    if len(set(all_labels)) > 1:  # Check if we have both classes
        f1 = f1_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        auroc = roc_auc_score(all_labels, all_probs)
    else:
        f1 = precision = recall = auroc = 0.0

    return f1, auroc, precision, recall

def comprehensive_evaluation(encoder, classifier, test_data):
    """Comprehensive evaluation for PhD research"""

    results = {
        'overall': {},
        'per_category': {},
        'summary': {}
    }

    print(f"\n📊 COMPREHENSIVE EVALUATION RESULTS:")
    print("="*60)

    # Overall evaluation
    all_test_images = []
    for category_images in test_data.values():
        all_test_images.extend(category_images)

    if len(all_test_images) > 0:
        simple_aug = SimpleAugmentation(config.IMAGE_SIZE)
        full_eval_dataset = Phase2Dataset(all_test_images, simple_aug)
        full_eval_loader = DataLoader(full_eval_dataset, batch_size=config.FINETUNE_BATCH_SIZE,
                                    shuffle=False, num_workers=2)

        overall_f1, overall_auroc, overall_precision, overall_recall = evaluate_model(
            encoder, classifier, full_eval_loader)

        results['overall'] = {
            'f1_score': overall_f1,
            'auroc': overall_auroc,
            'precision': overall_precision,
            'recall': overall_recall,
            'total_samples': len(all_test_images)
        }

        print(f"🏆 OVERALL PERFORMANCE:")
        print(f"   F1 Score: {overall_f1:.4f}")
        print(f"   AUROC: {overall_auroc:.4f}")
        print(f"   Precision: {overall_precision:.4f}")
        print(f"   Recall: {overall_recall:.4f}")
        print(f"   Target Achieved: {'✅' if overall_f1 > config.TARGET_F1_SCORE else '❌'}")

    # Save results
    try:
        results_path = os.path.join(config.SAVE_DIR, "evaluation", "results",
                                  "comprehensive_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"💾 Results saved to: comprehensive_results.json")
    except Exception as e:
        print(f"⚠️ Error saving results: {e}")

    return results

# ======================== MAIN EXECUTION ========================
def main():
    """Main execution function for PhD research"""

    print("🎓 STARTING PhD-LEVEL SIMCLR RESEARCH")
    print(f"🎯 Objective: Achieve F1 > {config.TARGET_F1_SCORE:.0%} with {config.LABELED_DATA_RATIO:.0%} labeled data")

    start_time = datetime.now()
    print(f"⏱️ Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Setup directories
        if not create_directory_structure():
            raise RuntimeError("Failed to create directory structure")

        # Discover data
        train_data, test_data, category_stats = discover_mvtec_data()

        print(f"\n📋 RESEARCH DATASET SUMMARY:")
        print(f"   Categories: {len(config.active_categories)}/{len(config.ALL_CATEGORIES)}")
        print(f"   Phase 1 (Pretraining): {sum(len(imgs) for imgs in train_data.values()):,} normal images")
        print(f"   Phase 2 (Evaluation): {sum(len(imgs) for imgs in test_data.values()):,} test images")

        # PHASE 1: Self-supervised pretraining
        print(f"\n🚀 EXECUTING PHASE 1: Self-Supervised Pretraining")
        pretrained_model, phase1_metrics = phase1_pretrain(train_data)

        # PHASE 2: Supervised fine-tuning and evaluation
        print(f"\n🚀 EXECUTING PHASE 2: Supervised Fine-tuning & Evaluation")
        classifier, phase2_metrics, final_results = phase2_finetune_and_evaluate(pretrained_model, test_data)

        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n" + "="*80)
        print("🎊 PhD RESEARCH COMPLETED SUCCESSFULLY!")
        print("="*80)

        print(f"⏱️ Total Duration: {duration}")
        print(f"🏆 Best F1 Score: {phase2_metrics.best_f1:.4f}")
        print(f"🎯 Target Achieved: {'✅ YES' if phase2_metrics.best_f1 > config.TARGET_F1_SCORE else '❌ NO'}")
        print(f"📊 Categories Evaluated: {len(config.active_categories)}")
        print(f"💡 Labeled Data Used: {config.LABELED_DATA_RATIO:.0%}")

        # Save final research summary
        research_summary = {
            'title': 'PhD Research: SimCLR for Industrial Anomaly Detection',
            'completion_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': str(duration),
            'target_f1': config.TARGET_F1_SCORE,
            'achieved_f1': phase2_metrics.best_f1,
            'target_achieved': phase2_metrics.best_f1 > config.TARGET_F1_SCORE,
            'labeled_data_ratio': config.LABELED_DATA_RATIO,
            'categories_evaluated': len(config.active_categories),
            'phase1_epochs': config.PRETRAIN_EPOCHS,
            'phase2_epochs': config.FINETUNE_EPOCHS,
            'final_results': final_results
        }

        summary_path = os.path.join(config.SAVE_DIR, "PhD_Research_Summary.json")
        with open(summary_path, 'w') as f:
            json.dump(research_summary, f, indent=2)

        print(f"📋 Final summary saved: PhD_Research_Summary.json")

        if phase2_metrics.best_f1 > config.TARGET_F1_SCORE:
            print(f"\n🎓 CONGRATULATIONS! Your PhD research target has been achieved!")
            print(f"🌟 F1 Score of {phase2_metrics.best_f1:.4f} exceeds the {config.TARGET_F1_SCORE:.0%} threshold")
        else:
            print(f"\n📈 Research Progress: {phase2_metrics.best_f1:.4f}/{config.TARGET_F1_SCORE:.4f}")
            print(f"💡 Consider: longer training, hyperparameter tuning, or data augmentation")

        return research_summary

    except KeyboardInterrupt:
        print(f"\n⚠️ Research interrupted by user")
        print(f"💾 Partial results may be available in {config.SAVE_DIR}")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print(f"\n🔧 TROUBLESHOOTING:")
        print(f"   - Verify DATA_ROOT: {config.DATA_ROOT}")
        print(f"   - Check Google Drive mount")
        print(f"   - Ensure sufficient GPU memory")
        print(f"   - Validate MVTec-AD dataset structure")
        raise

    finally:
        # Cleanup
        gc.collect()
        torch.cuda.empty_cache()
        print(f"🧹 Memory cleanup completed")

if __name__ == "__main__":
    main()