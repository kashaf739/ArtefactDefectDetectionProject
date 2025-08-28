# Industrial Anomaly Detection System

A Flask web application for detecting anomalies in industrial products using two deep learning approaches: CNN (Convolutional Neural Network) and SimCLR (Simple Contrastive Learning of Visual Representations). The system allows users to upload images and receive side-by-side analysis from both models, including confidence scores and error classification.

## Features

- **Dual Model Analysis**: Compare results from both CNN and SimCLR models
- **Error Classification**: Detailed identification of defect types with severity levels
- **Confidence Scoring**: Visual indicators for each model's prediction confidence
- **Modern UI**: Clean, responsive interface with intuitive design
- **Real-time Processing**: Fast image analysis with instant results
- **Recommendations**: Actionable insights based on detected defects

## Prerequisites

- Python 3.8 or higher
- PyTorch with CUDA support (recommended for GPU acceleration)
- Flask web framework
- Additional Python libraries (see Installation section)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/industrial-anomaly-detection.git
cd industrial-anomaly-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask pillow numpy albumentations
```

### 4. Model Files

Place your trained model files in the `models` directory:

- `models/model_epoch_3.pth` - CNN model
- `models/phase1_epoch_007.pth` - SimCLR encoder
- `models/phase2_epoch_030.pth` - SimCLR classifier

## Project Structure

```
industrial-anomaly-detection/
├── app.py                    # Main Flask application
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── models/                  # Model files directory
│   ├── model_epoch_3.pth    # CNN model
│   ├── phase1_epoch_007.pth  # SimCLR encoder
│   └── phase2_epoch_030.pth  # SimCLR classifier
├── templates/               # HTML templates
│   ├── index.html          # Home page
│   └── result.html         # Results page
└── uploads/                # Temporary file storage (created automatically)
```

## Running the Application

### 1. Start the Flask Server

```bash
python app.py
```

### 2. Access the Application

Open your web browser and navigate to:
- Local development: `http://localhost:8080`
- If running on Codespace: Use the provided forwarded port URL

### 3. Using the Application

1. **Upload an Image**:
   - Drag and drop an image onto the upload area
   - Or click "Browse Files" to select an image manually

2. **Analyze the Image**:
   - Click the "Analyze Image" button
   - Wait for the processing to complete

3. **View Results**:
   - Compare predictions from both CNN and SimCLR models
   - Review confidence scores for each prediction
   - Examine error classifications (if defects are detected)
   - Consider the provided recommendations

## API Endpoints

- `GET /`: Home page with image upload interface
- `POST /predict`: Process uploaded image and return results

## Model Architecture

### CNN Model
- **Backbone**: ResNet-50 pre-trained on ImageNet
- **Classification Head**: Custom layers with dropout and batch normalization
- **Training Approach**: Supervised learning with labeled data
- **Input**: 224x224 RGB images
- **Output**: Binary classification (Normal/Defective)

### SimCLR Model
- **Phase 1**: Self-supervised pretraining with contrastive learning
  - **Encoder**: ResNet-50 without the final layer
  - **Projection Head**: Maps features to lower-dimensional space
- **Phase 2**: Supervised fine-tuning with limited labeled data
  - **Classifier**: Linear layer for binary classification
- **Input**: 224x224 RGB images
- **Output**: Binary classification (Normal/Defective)

## Error Classification

When defects are detected, the system provides detailed error classification:

### CNN Model Error Types
- **Surface Scratch**: Visible surface damage with irregular patterns
- **Contamination**: Foreign particles or residue on surface

### SimCLR Model Error Types
- **Crack**: Linear fracture in material structure
- **Deformation**: Structural shape deviation from specification

### Severity Levels
- **High**: Critical defects requiring immediate attention
- **Medium**: Defects that should be addressed soon
- **Low**: Minor defects that should be monitored

## Dependencies

- Flask==2.3.3
- torch==2.0.1
- torchvision==0.15.2
- Pillow==10.0.0
- numpy==1.24.3
- albumentations==1.3.1

## Troubleshooting

### Model Loading Issues

If you encounter errors loading the models:

1. Verify that all model files exist in the `models` directory
2. Check that the model filenames match those specified in `app.py`
3. Ensure you have sufficient RAM/VRAM to load the models
4. For PyTorch 2.6 compatibility issues, the code includes fallback mechanisms

### Image Upload Issues

If you encounter problems with image uploads:

1. Check that the image format is supported (JPEG, PNG, BMP)
2. Ensure the file size is under 16MB
3. Verify that the `uploads` directory has write permissions

### Performance Issues

If the application runs slowly:

1. Ensure CUDA is available and properly configured:
   ```python
   print(torch.cuda.is_available())
   ```
2. Close unnecessary applications to free up system resources
3. Consider reducing image resolution for faster processing

## Future Enhancements

- Enhanced model inclusion for real-world industry level deployment
- Support for additional model architectures
- Batch image processing capability
- User authentication and history tracking
- Export functionality for results
- Integration with industrial IoT systems
- Real-time video stream analysis

## License

This is an academic work for university project only.

## Contact

Wajeeha Kashaf - 2404372@leedstrinity.ac.uk

## Acknowledgments

- MVTec AD Dataset for providing the industrial anomaly detection dataset
- PyTorch team for the deep learning framework
- Flask community for the web framework
- Bootstrap for the UI components