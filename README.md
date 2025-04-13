# FacePredictor

# Webcam Face Prediction - Gender, Age, and Emotion Detection

This project is a Streamlit-based application that uses a webcam to detect and predict the gender, age, and emotion of individuals in real-time. The application uses deep learning models pre-trained on large datasets and fine-tuned for gender, age, and emotion classification. The predictions are made using a combination of the EfficientNet, VGG19, and Mediapipe libraries.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Models](#models)
4. [Dependencies](#dependencies)
5. [How to Use](#how-to-use)
6. [Setup and Installation](#setup-and-installation)
7. [Acknowledgments](#acknowledgments)

---

## Overview

This application leverages the power of machine learning models to detect three key attributes of a person’s face in real-time:
- **Gender**: Male or Female
- **Age**: Estimated age of the person
- **Emotion**: Classifies emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

The Streamlit interface enables users to capture webcam images and see real-time predictions with annotated gender, age, and emotion labels directly on the image.

In addition to the main **Streamlit application**, there is also a **Jupyter Notebook** that demonstrates how to use the same deep learning models for gender, age, and emotion prediction directly from the webcam, **without the need for Streamlit**.

---

## Features

- **Webcam Integration**: Captures images directly from your webcam for real-time face analysis.
- **Gender Detection**: Predicts whether the face corresponds to a male or female based on the image.
- **Age Prediction**: Estimates the person’s age based on their facial features.
- **Emotion Recognition**: Classifies the facial expression into one of the seven basic emotions.
- **Real-Time Processing**: Predictions are updated live as the webcam captures the image.

---

## Models

This application uses three pre-trained and fine-tuned models:

### 1. **Gender Model**:
- **Base Model**: EfficientNet-B0
- **Fine-Tuned For**: Gender prediction (Male/Female)
- **Layers Modified**: Classifier layers were replaced and fine-tuned to predict gender from facial features.
- **Dataset**: [UTKFace dataset](https://susanqq.github.io/UTKFace/)

### 2. **Age Model**:
- **Base Model**: VGG19
- **Fine-Tuned For**: Age prediction (regression task to predict the exact age)
- **Layers Modified**: The final classifier layers were replaced with a custom architecture to predict the age.
- **Dataset**: [UTKFace dataset](https://susanqq.github.io/UTKFace/)

### 3. **Emotion Model**:
- **Base Model**: VGG19
- **Fine-Tuned For**: Emotion classification (7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Layers Modified**: Classifier layers were modified to predict the emotions from the facial features.
- **Dataset**: [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

The models are fine-tuned using pre-trained weights and are loaded into the application for inference.

---

## Dependencies

To run this application, the following Python libraries are required:

- `streamlit`
- `torch`
- `torchvision`
- `mediapipe`
- `opencv-python`
- `Pillow`
- `numpy`
- `matplotlib`

You can install the necessary dependencies using pip:

    
      pip install streamlit torch torchvision mediapipe opencv-python Pillow nump matplotlib
      
## How to Use

### Run the Application:
- Clone or download this repository to your local machine.
- Install the dependencies as mentioned in the [Dependencies](#dependencies) section.
- Open the terminal and run the following command to start the Streamlit application:
  
  ```bash
  streamlit run FacePredictor_app.py
  
- This will open a local server, and you can access the application through your browser.

### Interact with the Webcam:
- Once the app is running, you will see a live webcam feed displayed in the browser.
- The app will automatically detect faces, and predictions for gender, age, and emotion will be shown with annotated text.

### Capture and Display Predictions:
- The application will highlight the detected face(s) and display real-time predictions for gender, age, and emotion.


### Run the Application (Without Streamlit, Using Jupyter Notebook):

In addition to the main Streamlit application, this project also includes a Jupyter Notebook that allows you to use the webcam for gender, age, and emotion predictions without using Streamlit. To use this notebook:

1.  Ensure you have all the required dependencies installed as mentioned in the [Dependencies](#dependencies) section.
    
2.  Open the **FacePredictor_app.ipynb** file from app directory in Jupyter Notebook or JupyterLab.
    
3.  Run the cells sequentially to start the webcam and see real-time predictions for gender, age, and emotion directly.
    

This notebook performs the same functionality as the Streamlit app but in a notebook interface. It uses OpenCV to access the webcam and the models to make predictions.

## Setup and Installation

1.  **Clone the Repository**
    
	    git clone https://github.com/ParsaKhmd/FacePredictor.git
		cd FacePredictor

2. **Run the Application**: Start the Streamlit app with:	
	
		streamlit run FacePredictor_app.py
		
### 📥 Model Weights

This application requires three fine-tuned PyTorch model weights. Due to their large size, they are **not included in the repository**. You can download them from the links below and place them in the **root directory** of the project.

| Model                    | File Name                      | Download Link |
|--------------------------|--------------------------------|----------------|
| Gender Prediction Model  | `fine_tuned_gender_model.pth`  | [Download](https://drive.google.com/drive/folders/12GnZIXySo3p3KtZfxs9XUkd-WLXI7fAh?usp=sharing)  |
| Age Prediction Model     | `fine_tuned_age_model.pth`     | [Download](https://drive.google.com/drive/folders/1z3bun3mhV9AAC0FsfDibIX0WnS2h3GIu?usp=sharing)  |
| Emotion Recognition Model| `emotion_model_fine_tuned.pth` | [Download](https://drive.google.com/drive/folders/1AcNPYxRBX-YTpB38tjNcYQ73xyn66jN6?usp=sharing)  |

> 🔔 **Note**: These models must be placed in the **same directory** as `FacePredictor_app.py` (or update the path accordingly in the code).
	
	
## Acknowledgments

- **EfficientNet-B0** and **VGG19** models are pre-trained on ImageNet and are 			available through PyTorch’s `torchvision` library.
- **Mediapipe** is used for real-time face detection and has been integrated for accurate face localization in the webcam feed.
- This project is built with **Streamlit** to provide an interactive user interface for real-time predictions.
