import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load your trained model
print("Loading model...")
model = tf.keras.models.load_model(
    "C:/plant-disease-ai/models/plant_disease_model.h5"
)

# Load class names
with open("C:/plant-disease-ai/models/class_names.json", "r") as f:
    class_names = json.load(f)

print(f"Model loaded! Can detect {len(class_names)} diseases!")

# Prediction function
def predict_disease(image):
    # Resize image to 224x224
    img = image.resize((224, 224))
    
    # Convert to array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get prediction
    predictions = model.predict(img_array)
    
    # Get top 3 predictions
    top3_indices = np.argsort(predictions[0])[-3:][::-1]
    
    results = {}
    for idx in top3_indices:
        disease = class_names[idx].replace("___", " - ").replace("_", " ")
        confidence = float(predictions[0][idx]) * 100
        results[disease] = confidence
    
    return results

# Build the Gradio app
app = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil", label="Upload a leaf photo"),
    outputs=gr.Label(num_top_classes=3, label="Disease Prediction"),
    title="🌿 Plant Disease Detector",
    description="Upload a leaf photo and AI will detect the disease!",
    examples=[],
    theme=gr.themes.Soft()
)

print("Starting app...")
app.launch()