import streamlit as st
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
import mediapipe as mp

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write(f"Using device: {device}")

# -- Load Gender Model
gender_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
gender_model = torchvision.models.efficientnet_b0(weights=gender_weights).to(device)

for param in gender_model.parameters():
    param.requires_grad = True
    
gender_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(1280, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1)

).to(device)

# -- Load age Model
age_weights = torchvision.models.VGG19_Weights.DEFAULT
age_model = torchvision.models.vgg19( weights=age_weights)

# Freeze all parameters in the feature layers
for param in age_model.features.parameters():
    param.requires_grad = False

# Unfreeze only the last 20 layers of the feature layers
for param in age_model.features[-20:].parameters():
    param.requires_grad = True


age_model.classifier = nn.Sequential(
    
    nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(0.5),  
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU()
    ),

    
    nn.Sequential(
        nn.Linear(4096, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.4), 
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        
    ),

   
    nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),  
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),  
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        
    ),

   
    nn.Linear(256, 1)
).to(device)

# -- Load emotion Model
emotion_weights = torchvision.models.VGG19_Weights.DEFAULT
emotion_model = torchvision.models.vgg19( weights=emotion_weights)

# Freeze all parameters in the feature layers
for param in emotion_model.features.parameters():
    param.requires_grad = False
    
# Unfreeze only the last 20 layers of the feature layers
for param in emotion_model.features[-20:].parameters():
    param.requires_grad = True    


# Modify the classifier
emotion_model.classifier = nn.Sequential(
    
    nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(4096, 4096),
        nn.BatchNorm1d(4096),
        nn.ReLU()
    ),

    
    nn.Sequential(
        nn.Linear(4096, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(1024, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        
    ),

    
    nn.Sequential(
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
       
    ),

    
    nn.Linear(256, 7)  # 7 emotion classes
).to(device)




gender_model.load_state_dict(torch.load("fine_tuned_gender_model.pth", map_location=device))
gender_model.to(device).eval()

age_model.load_state_dict(torch.load("fine_tuned_age_model.pth", map_location=device))
age_model.to(device).eval()

emotion_model.load_state_dict(torch.load("emotion_model_fine_tuned.pth", map_location=device))
emotion_model.to(device).eval()

# Data transformation for gender model
gender_infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data transformation for age model
age_infer_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data transformation for emotion model
emotion_infer_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# Helper function to predict and display results
def predict_and_display(frame, face_box, gender_model, age_model, emotion_model):
    x, y, w, h = face_box
    face = frame[y:y+h, x:x+w]

    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    # Process the face image for each model
    gender_tensor = gender_infer_transform(face_pil).unsqueeze(0).to(device)
    age_tensor = age_infer_transform(face_pil).unsqueeze(0).to(device)
    emotion_tensor = emotion_infer_transform(face_pil).unsqueeze(0).to(device)

    # Predictions
    with torch.no_grad():
        gender_output = gender_model(gender_tensor)
        age_output = age_model(age_tensor)
        emotion_output = emotion_model(emotion_tensor)

        # Postprocess predictions
        gender_pred = torch.round(torch.sigmoid(gender_output))
        gender_index = int(gender_pred.item())
        gender_labels = ['Male', 'Female']
        gender_text = gender_labels[gender_index]

        age_text = int(age_output.item())

        emotion_index = torch.argmax(emotion_output, dim=1).item()
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        emotion_text = emotion_labels[emotion_index]

    # Annotate the frame with the predictions
    label = f"Gender: {gender_text}, {age_text} years old, Emotion: {emotion_text}"
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # Adjust the position to the left side (for example, at the coordinates (10, 30))
    cv2.putText(frame, label, (10, 30),  # x=10, y=30 puts the text on the left
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# Streamlit app structure
st.title("Webcam Face Prediction")
st.sidebar.write("This app uses your webcam to detect gender, age, and emotion.")

camera_input = st.camera_input("Capture Image")

if camera_input:
    # Convert the camera input (PIL image) to OpenCV format (numpy array)
    img = Image.open(camera_input)
    img = np.array(img)

    # Initialize face detection model from Mediapipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
        # Convert the image to RGB for face detection
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        # Process face detections
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * img.shape[1])
                y = int(bbox.ymin * img.shape[0])
                w = int(bbox.width * img.shape[1])
                h = int(bbox.height * img.shape[0])

                # Clamp values to image size
                x, y = max(0, x), max(0, y)
                w, h = min(w, img.shape[1] - x), min(h, img.shape[0] - y)

                # Make predictions and display
                img = predict_and_display(img, (x, y, w, h), gender_model, age_model, emotion_model)

    # Display the image with the annotations
    st.image(img, caption="Predicted Image", channels="BGR")
