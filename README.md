MSc-Final-Project

Exploring the Potential of Self-Supervised Learning for Industrial Defect Detection

# MVTec-AD Anomaly Detection: CNN (Supervised)(classical) and SimCLR (SSL)(linear classifier 10% labelled data) Approaches

## Project Overview

This project implements two state-of-the-art approaches for industrial anomaly detection on the MVTec Anomaly Detection (MVTec-AD) dataset:

1. **Supervised CNN Approach**: A traditional ResNet-50 based model 
2. **Self-Supervised SimCLR Approach**: A two-phase framework using contrastive learning with linear classifier (limited labeled data)

The project aims SSL models achieving F1 ≥ 75% with minimal labelled data (initial
hope), making it suitable for industrial applications where labeled anomalies are scarce.

Aims and SMART Objectives
Primary Aim:
To evaluate the performance of SimCLR in learning defect-relevant features from
unlabelled industrial images.
SMART Objectives:
• Pretrain SimCLR on unlabelled images from MVTec AD
• Fine-tune a linear classifier on 5–10% labelled defect data
• Build and train a supervised CNN as a performance baseline
• Evaluate models using Accuracy, Precision, Recall, and F1 Score
• Visualize embeddings with t-SNE and interpretability tools like Grad-CAM or other
• Define success as SSL models achieving F1 ≥ 75% with minimal labelled data (initial
hope)
Research Questions
1. 2. 3. Can SimCLR learn effective visual features from unlabelled industrial
data?
How do SSL models compare with supervised models trained on labelled datasets?
What are the limitations or operational boundaries where SSL is viable in industrial
(real-world) QA?
Artefact Deliverables
• A functional SSL pipeline (SimCLR)
• Evaluation reports and accuracy/F1-score metrics
• Visualization outputs
• Final written report documenting methodology, findings, results including diagrams

## Dataset

The MVTec-AD dataset is a comprehensive for industrial anomaly detection, containing:
- **15 categories** of industrial products (bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper)
- **5,354 high-resolution images** (3,629 for training, 1,725 for testing)
- **Various defect types**: Scratches, dents, contamination, structural defects, etc.

### Dataset Structure
```
mvtec_ad/
├── bottle/
│   ├── train/good/
│   └── test/
│       ├── good/
│       ├── broken_large/
│       ├── broken_small/
│       └── contamination/
├── cable/
├── capsule/
└── ... (other categories)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Google Drive (for Colab implementation)

### Setup
1. Clone the repository:
```bash
git clone https:
cd mvtec-ad-anomaly-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the MVTec-AD dataset:
```bash
# Download from official website: https://www.mvtec.com/company/research/datasets/mvtec-ad
# Extract and place in your Google Drive or local directory
```

4. Configure paths in the scripts:
   - Update `DATASET_ROOT` in both `train_cnn.py` and `train_simclr.py`
   - Update `OUTPUT_DIR` for model and results storage

## Project Structure

```
├──src/
    ├─ train_cnn.py              # CNN training script
    ├── evaluate_cnn.py           # CNN evaluation script
    ├── train_simclr.py           # SimCLR training script
    └── evaluate_simclr.py        # SimCLR evaluation script
├── requirements.txt          # Python dependencies
├── README.md                 # this file
├── app/
├   ├── models/                   # Saved models for app 
│   ├── cnn/
│   └── simclr/
├── results/                  # Evaluation results and visualizations
│   ├── self_supervised/
│   └── supervised/
└── data/                     # Dataset (not included)
    └── mvtec_ad/
```

## CNN Approach

The CNN approach uses a supervised learning paradigm with a ResNet-50 backbone modified for binary anomaly detection.

### Architecture
- **Backbone**: ResNet-50 pre-trained on ImageNet
- **Modifications**:
  - Unfreeze last two residual blocks (layer3, layer4) for fine-tuning
  - Replace final FC layer with custom head:
    - Dropout (0.5)
    - Linear (2048 → 512)
    - ReLU + BatchNorm
    - Dropout (0.3)
    - Linear (512 → 2)
- **Loss**: Weighted Cross-Entropy Loss (handles class imbalance)
- **Optimizer**: AdamW with OneCycleLR scheduler

### Training (CNN)

#### Key Features
- **Data Augmentation**: Advanced transformations using Albumentations
- **Class Balancing**: Automatic weight calculation for imbalanced datasets
- **Early Stopping**: Patience-based stopping to prevent overfitting
- **Checkpointing**: Save best model and periodic checkpoints

#### Training Command
```bash
python train_cnn.py
```

#### Configuration
Adjust these parameters in `Config` class:
```python
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.2
PATIENCE = 5  # Early stopping
```

### Evaluation (CNN)

#### Comprehensive Metrics
- **Primary Metrics**: Accuracy, F1-Score, Precision, Recall
- **Advanced Metrics**: ROC-AUC, PR-AUC, Specificity
- **Confusion Matrix**: Detailed classification analysis

#### Evaluation Command
```bash
python evaluate_cnn.py
```

#### Key Features
- **Model Loading**: Automatic loading of best checkpoint
- **Failure Analysis**: Detailed misclassification analysis
- **Visualizations**: ROC curves, confusion matrices, t-SNE plots
- **Grad-CAM**: Model interpretability through attention visualization

## SimCLR Approach

The SimCLR approach implements a two-phase self-supervised learning framework designed to work with limited labeled data.

### Architecture
- **Phase 1 (Self-Supervised)**:
  - **Encoder**: ResNet-50 backbone
  - **Projection Head**: MLP (2048 → 1024 → 256)
  - **Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Phase 2 (Supervised Fine-tuning)**:
  - **Frozen Encoder**: Uses representations from Phase 1 epoch  7
  - **Linear Classifier**: Simple logistic regression on top of features
  - **Data Efficiency**: Uses only 10% labeled data for fine-tuning

### Training (SimCLR)

#### Phase 1: Self-Supervised Pretraining
- **Objective**: Learn representations without labels
- **Data**: Only "good" samples from training set
- **Augmentation**: Strong augmentations for contrastive learning
- **Temperature**: 0.07 (optimal for contrastive loss)

#### Phase 2: Supervised Fine-tuning
- **Objective**: Train classifier on limited labeled data
- **Data**: 10% of test set (stratified sampling)
- **Efficiency**: Minimal labeled data requirement

#### Training Command
```bash
python train_simclr.py
```

#### Configuration
Key parameters in `Config` class:
```python
# Phase 1
PRETRAIN_EPOCHS = 7
PRETRAIN_BATCH_SIZE = 48
FEATURE_DIM = 256
TEMPERATURE = 0.07

# Phase 2
FINETUNE_EPOCHS = 30
FINETUNE_BATCH_SIZE = 32
LABELED_DATA_RATIO = 0.1  # 10% labeled data
```

### Evaluation (SimCLR)

#### Evaluation Command
```bash
python evaluate_simclr.py
```

#### Key Features
- **Comprehensive Metrics**: Same as CNN approach
- **Representation Analysis**: t-SNE visualizations of learned features
- **Error Analysis**: Detailed breakdown of false positives/negatives
- **Model Comparison**: Comparative analysis with CNN approach only in Dissertation report
- **Efficiency Metrics**: Performance vs. labeled data usage

## Results and Metrics

### Performance Comparison

| Approach       | Accuracy | F1-Score | Precision | Recall | Labeled Data Used |
|----------------|----------|----------|-----------|--------|---------|---------|
| CNN            | 92.63    | 80.81    | 89.55     | 73.62  | 100%              |
| SimCLR (10%)   | 72.75    | 84.18    | 73.01     | 99.36  | 10%               |

### Key Findings

## Visualizations

### CNN Visualizations
1. **Confusion Matrix**: Detailed classification performance
2. **ROC/PR Curves**: Threshold-independent performance analysis
3. **Grad-CAM**: Attention maps showing model focus areas
5. **t-SNE Plots**: Feature space visualization

### SimCLR Visualizations
1. **Feature Space**: t-SNE plots of learned representations
2. **Workflow Diagram**: Two-phase training methodology
3. **Failure Analysis**: Per-category error breakdown
4. **Grad-CAM**: Attention maps showing model focus areas
5. **Confusion Matrix**: Detailed classification performance


### Example Visualizations
- **Grad-CAM**: Shows model attention on defective regions
- **t-SNE**: Demonstrates separation of normal/anomalous samples
- **Confusion Matrix**: Reveals class-specific performance patterns
- **ROC Curves**: Illustrates trade-off between sensitivity and specificity

## Key Features

### CNN Approach Features
1. **Robust Architecture**: Modified ResNet-50 with regularization
2. **Advanced Augmentation**: Albumentations pipeline for improved generalization 
3. **Class Imbalance Handling**: Weighted loss and sampling strategies still biased towards good/normal
4. **Comprehensive Evaluation**: Multiple metrics and failure analysis
5. **Interpretability**: Grad-CAM visualizations for model transparency

### SimCLR Approach Features
1. **Self-Supervised Learning**: Reduces labeled data dependency
2. **Two-Phase Training**: Efficient representation learning
3. **Data Efficiency**: Achieves desired F1 score with 10% labels but biased towards defected/anomalous
4. **Representation Analysis**: t-SNE and feature space examination

### Shared Features
1. **Modular Design**: Easy to modify and extend
2. **Comprehensive Metrics**: Multiple evaluation perspectives
3. **Visualization Suite**: Rich set of analysis tools
4. **GPU Optimization**: Efficient training and evaluation
5. **Checkpointing**: good model saving and loading

###Strengths & Limitations
Strengths:
•	Lightweight training on Colab
•	Visual explainability included
Limitations:
•	The limitations of this study include computational constraints during pretraining, the binary classification approach that may oversimplify complex defect scenarios, and the focus on a single SSL architecture (SimCLR). Future research should explore more efficient pretraining strategies, investigate hybrid SSL-supervised approaches, and evaluate the performance of other SSL architectures like BYOL in industrial settings.
•	Sensitive to augmentation parameters and configurations and temperatures


Future Work
•	Use different dataset and instead of linear classifier use SVM or Mahalnobis (techniques) for getting unbiased results
•	Explore more stronger augmentations.
•	Develop knowledge distillation for edge-deployable lightweight models
•	Incorporate semi-supervised fine-tuning after pretraining such as the use of SVM a supervised technique which is computationally less heavy and don’t take longer to execute. On top of that SVM can give excellent and robust performance when labelled data is scarse as compared to a linear classier.
•	Generate synthetic defects via GANs/diffusion models for balanced training for datasets like MVTEC



## License

This project is part of a university MSc program and is intended for academic and educational use only.

## Acknowledgments

- MVTec for providing the Anomaly Detection dataset
- PyTorch team for the excellent deep learning framework
- Albumentations team for the comprehensive augmentation library
- Scikit-learn team for essential machine learning utilities

## Contact

For questions or suggestions, please contact:
- Wajeeha Kashaf
- wajeehakashaf7391@gmail.com
- This README was generated with the assistance of ChatGPT.
