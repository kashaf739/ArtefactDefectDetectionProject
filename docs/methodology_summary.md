# Methodology Summary

## Dataset: MVTec Anomaly Detection

The MVTec Anomaly Detection (MVTec AD) dataset was selected as the foundation for this research due to its unparalleled suitability for industrial defect detection tasks. Key reasons for its selection include industrial relevance (real-world manufacturing defects across diverse categories), comprehensive annotations (pixel-level ground truth masks), methodological alignment with self-supervised learning (training set contains exclusively defect-free samples), and defect diversity (structural, textural, and logical defects mirroring real-world challenges).

### Dataset Structure and Content
The dataset follows a hierarchical structure with 15 industrial object categories (bottle, cable, capsule, metal-nut, screw, etc.), each containing 5-15 defect types. Key specifications include:
- High-resolution images (typically 1024×1024 pixels)
- Pixel-level binary masks for defective regions
- Sample distribution: ~60-400 defect-free training images per category; ~60-150 test images per category (balanced between good and defective samples)

This structure ensures diversity in material properties (transparent, metallic, flexible), object scales (small screws to larger bottles), and defect characteristics.

### Dataset Challenges
The MVTec AD dataset presents several significant challenges for anomaly detection models:
- Zero-Shot Anomaly Detection: Training exclusively on defect-free samples requires models to learn normality representations without prior exposure to anomalies
- Subtle Defects: Many defects are barely perceptible (e.g., faint scratches, slight color variations)
- Category-Specific Normality: Each category exhibits unique normal variations
- Defect Heterogeneity: Defects vary significantly in size, shape, and appearance within categories
- Real-World Complexity: Includes reflections, shadows, and manufacturing variations that could be mistaken for defects

### Data Preparation
For computational efficiency on Google Colab, preprocessing steps included:
1. Resizing images to 256×256 pixels while maintaining aspect ratio
2. Normalization using ImageNet mean and standard deviation
3. Data augmentation (random horizontal flipping and color jittering) applied to training images

This preparation ensured optimal model performance while accommodating computational constraints and maintaining the dataset's original characteristics.

## Supervised CNN Baseline Approach

The supervised methodology employed a ResNet-50 architecture pre-trained on ImageNet for anomaly detection. Data preprocessing involved systematic scanning and labeling of images as "good" (0) or "defective" (1), with stratified 80/20 train-validation split. A comprehensive augmentation pipeline using Albumentations was implemented for training (horizontal/vertical flips, rotation, color jittering, Gaussian blur, resizing, normalization) while validation used only resizing and normalization.

The model architecture featured selective fine-tuning of layers 3 and 4, with other layers frozen. The original classifier was replaced with a custom design including dropout (0.5), ReLU activation, batch normalization, and second dropout layer (0.3). Training utilized AdamW optimizer (lr=0.001, weight decay=1e-4), OneCycleLR scheduler (max lr=0.01), weighted cross-entropy loss, batch size 64, and early stopping (patience=5 epochs). Evaluation metrics included training/validation loss, accuracy, F1-score, and confusion matrices.

## Self-Supervised Learning (SimCLR) Approach

The SSL methodology implemented a two-phase training process. Phase 1 involved pretraining SimCLR on only "good" images using ResNet-50 backbone with projection head (2048→1024→256 dimensions), NT-Xent loss (temperature=0.07), batch size 48, and 7 epochs. Phase 2 froze the pretrained encoder and added a linear classifier (2048→2), trained on 10% labeled data with Adam optimizer (lr=1e-4), batch size 32, and 30 epochs.

Data preprocessing included validation of file extensions, size filtering, dimension checks, and corrupted image handling. Phase 1 used only "good" training images, while Phase 2 sampled 10% labeled test data. The implementation disabled mixed precision training, enabled TF32 computation, and used gradient clipping (max norm=1.0). Visualization capabilities tracked training loss, learning rate schedule, F1-score progression, and AUROC evolution.

## Evaluation Results

The supervised CNN achieved 92.63% accuracy, 80.81% F1-score, 94.96% ROC-AUC, and 89.22% PR-AUC, with 97.70% recall on good samples and 73.62% recall on defective samples. The SSL SimCLR model achieved 84.18% F1-score with 99.36% recall but suffered from 98.93% false positive rate. Comparative analysis showed the supervised model had higher precision and specificity, while the SSL model had higher recall but extremely low specificity.

## Challenges and Solutions

Key challenges addressed included mixed-precision instability (resolved by using fp32), GPU memory constraints (managed through conservative batch sizes and gradient clipping), and validation computational cost (controlled through periodic validation with capped batches). The implementation incorporated class imbalance handling through weighted loss, early stopping, error handling for image loading, and systematic documentation.

## Contributions and Limitations

The research demonstrated SSL's potential to bridge supervised and unsupervised defect detection approaches, reducing labeling costs while maintaining strong performance. Limitations included computational constraints, binary classification approach, focus on single SSL architecture, and sensitivity to augmentation parameters. The SSL model showed excellent feature extraction capabilities but required optimization of the classification head.

## Future Work

Recommended future directions include exploring stronger augmentations, developing knowledge distillation for edge deployment, incorporating semi-supervised fine-tuning with SVM, and generating synthetic defects via GANs/diffusion models for balanced training.