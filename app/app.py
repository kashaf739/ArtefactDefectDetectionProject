import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import albumentations as A
from albumentations.pytorch import ToTensorV2
import base64
import warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "anomaly_detection_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to your models in codespace
CNN_MODEL_PATH = "models/model_epoch_3.pth"  # Update this path if needed
SIMCLR_ENCODER_PATH = "models/phase1_epoch_007.pth"  # Update this path if needed
SIMCLR_CLASSIFIER_PATH = "models/phase2_epoch_030.pth"  # Update this path if needed

# Load CNN model
try:
    # For PyTorch 2.6 compatibility, we need to handle the weights_only parameter
    try:
        cnn_checkpoint = torch.load(CNN_MODEL_PATH, map_location=device, weights_only=False)
    except:
        # If weights_only=False fails, try the default
        cnn_checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
    
    if isinstance(cnn_checkpoint, dict) and 'model_state_dict' in cnn_checkpoint:
        cnn_state_dict = cnn_checkpoint['model_state_dict']
    else:
        cnn_state_dict = cnn_checkpoint
    
    # Create a ResNet50 model
    cnn_model = models.resnet50(weights=None)
    
    # Check if the state_dict has the custom head structure
    if any(key.startswith('fc.') for key in cnn_state_dict.keys()):
        # It has the custom head, so we need to replace the fc layer
        num_ftrs = cnn_model.fc.in_features
        cnn_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 2)  # Binary classification
        )
    
    # Load the state dict
    cnn_model.load_state_dict(cnn_state_dict)
    cnn_model = cnn_model.to(device)
    cnn_model.eval()
    print("CNN model loaded successfully")
except Exception as e:
    print(f"Error loading CNN model: {e}")
    cnn_model = None

# Define a class to match the SimCLR model structure
class SimCLRModel(nn.Module):
    def __init__(self):
        super(SimCLRModel, self).__init__()
        # Encoder (ResNet50 without the final layer)
        self.encoder = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
        # Projection head (not used in inference)
        self.projector = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, 1)
        z = self.projector(h)
        return h, z

# Load SimCLR model
try:
    # For PyTorch 2.6 compatibility, we need to handle the weights_only parameter
    # and add numpy scalar to safe globals
    try:
        # First try with weights_only=False
        simclr_encoder_checkpoint = torch.load(SIMCLR_ENCODER_PATH, map_location=device, weights_only=False)
    except:
        # If that fails, try with safe globals
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        simclr_encoder_checkpoint = torch.load(SIMCLR_ENCODER_PATH, map_location=device)
    
    if isinstance(simclr_encoder_checkpoint, dict) and 'model_state_dict' in simclr_encoder_checkpoint:
        simclr_encoder_state = simclr_encoder_checkpoint['model_state_dict']
    else:
        simclr_encoder_state = simclr_encoder_checkpoint
    
    # Create the full SimCLR model
    simclr_model = SimCLRModel().to(device)
    
    # Load the state dict
    simclr_model.load_state_dict(simclr_encoder_state)
    simclr_model.eval()
    
    # Extract just the encoder part for inference
    simclr_encoder = simclr_model.encoder
    
    # Load the classifier
    try:
        simclr_classifier_checkpoint = torch.load(SIMCLR_CLASSIFIER_PATH, map_location=device, weights_only=False)
    except:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        simclr_classifier_checkpoint = torch.load(SIMCLR_CLASSIFIER_PATH, map_location=device)
    
    if isinstance(simclr_classifier_checkpoint, dict) and 'classifier_state_dict' in simclr_classifier_checkpoint:
        simclr_classifier_state = simclr_classifier_checkpoint['classifier_state_dict']
    else:
        simclr_classifier_state = simclr_classifier_checkpoint
    
    # Create the classifier
    simclr_classifier = nn.Linear(2048, 2).to(device)
    
    # Load the classifier state
    simclr_classifier.load_state_dict(simclr_classifier_state)
    simclr_classifier.eval()
    
    print("SimCLR model loaded successfully")
except Exception as e:
    print(f"Error loading SimCLR model: {e}")
    simclr_encoder = None
    simclr_classifier = None

# Define preprocessing transforms
cnn_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

simclr_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict_image(image_path):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    results = {
        'cnn': {
            'prediction': 'Model not loaded',
            'confidence': 0.0
        },
        'simclr': {
            'prediction': 'Model not loaded',
            'confidence': 0.0
        }
    }
    
    # CNN prediction
    if cnn_model is not None:
        try:
            image_np = np.array(image)
            cnn_tensor = cnn_transform(image=image_np)['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                cnn_outputs = cnn_model(cnn_tensor)
                cnn_probs = torch.nn.functional.softmax(cnn_outputs, dim=1)[0]
                cnn_confidence = cnn_probs[1].item() * 100  # Probability of defective
                cnn_prediction = "Defective" if cnn_probs[1] > cnn_probs[0] else "Normal"
                
                results['cnn'] = {
                    'prediction': cnn_prediction,
                    'confidence': cnn_confidence
                }
        except Exception as e:
            print(f"Error during CNN prediction: {e}")
    
    # SimCLR prediction
    if simclr_encoder is not None and simclr_classifier is not None:
        try:
            simclr_tensor = simclr_transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = simclr_encoder(simclr_tensor)
                features = torch.flatten(features, 1)
                simclr_outputs = simclr_classifier(features)
                simclr_probs = torch.nn.functional.softmax(simclr_outputs, dim=1)[0]
                simclr_confidence = simclr_probs[1].item() * 100  # Probability of defective
                simclr_prediction = "Defective" if simclr_probs[1] > simclr_probs[0] else "Normal"
                
                results['simclr'] = {
                    'prediction': simclr_prediction,
                    'confidence': simclr_confidence
                }
        except Exception as e:
            print(f"Error during SimCLR prediction: {e}")
    
    return results

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get predictions
        results = predict_image(filepath)
        
        # Convert image to base64 for display
        with open(filepath, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', results=results, filename=filename, image_data=encoded_string)
    
    return redirect(request.url)

# Error handlers
@app.errorhandler(413)
def too_large(e):
    flash('File too large. Maximum size is 16MB.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)