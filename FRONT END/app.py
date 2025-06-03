from flask import Flask, render_template, request
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2

# Initialize Flask app
app = Flask(__name__)

# Define paths
SAVED_IMAGES_FOLDER = "static/uploaded_images"
PREDICTED_IMAGES_FOLDER = "static/predicted_images"
MODEL_PATH = "new_models/mobilenet_model.pth"

# Ensure directories exist
os.makedirs(SAVED_IMAGES_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_IMAGES_FOLDER, exist_ok=True)

# Define preprocessing for MobileNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to preprocess a single image
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0)  # Add batch dimension

# Function to predict a single image (simplified to return only the label)
def predict_single_image(model, image_path, reverse_label_map):
    image_tensor = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, prediction = torch.max(outputs, 1)
        predicted_label = reverse_label_map[prediction.item()]
    return predicted_label

# Load model
num_classes = 14
loaded_mobilenet = models.mobilenet_v2(pretrained=False)
loaded_mobilenet.classifier[1] = nn.Linear(loaded_mobilenet.last_channel, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_mobilenet.load_state_dict(torch.load(MODEL_PATH, map_location=device))
loaded_mobilenet.to(device)
loaded_mobilenet.eval()

# Class mapping
reverse_label_map = {
    0: "XR_ELBOW abnormal",
    1: "XR_ELBOW Normal",
    2: "XR_FINGER abnormal",
    3: "XR_FINGER Normal",
    4: "XR_FOREARM abnormal",
    5: "XR_FOREARM Normal",
    6: "XR_HAND abnormal",
    7: "XR_HAND Normal",
    8: "XR_HUMERUS abnormal",
    9: "XR_HUMERUS Normal",
    10: "XR_SHOULDER abnormal",
    11: "XR_SHOULDER Normal",
    12: "XR_WRIST abnormal",
    13: "XR_WRIST Normal"
}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            file = request.files['file']
            if file.filename == '':
                return "Error: No file selected", 400
            if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                return "Error: Only .jpg, .jpeg, and .png files are allowed", 400

            filename = file.filename
            upload_path = os.path.join(SAVED_IMAGES_FOLDER, filename)
            file.save(upload_path)

            predicted_class = predict_single_image(loaded_mobilenet, upload_path, reverse_label_map)

            parts = predicted_class.split()
            bone_type = " ".join(parts[:-1])
            status = parts[-1].capitalize()

            predicted_image_path = os.path.join(PREDICTED_IMAGES_FOLDER, 'predicted_' + filename)
            cv2.imwrite(predicted_image_path, cv2.imread(upload_path))

            return render_template(
                'prediction.html',
                status=status,
                bone_type=bone_type,
                file_name=filename,
                prediction_image='predicted_images/predicted_' + filename
            )

        except Exception as e:
            print("Prediction Error:", str(e))
            return render_template('prediction.html', error=f"Error processing file: {str(e)}"), 500

    return render_template('prediction.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)