import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# ── Load model ──────────────────────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(
    "C:/plant-disease-ai/models/plant_disease_model.h5"
)

with open("C:/plant-disease-ai/models/class_names.json", "r") as f:
    class_names = json.load(f)

print(f"Model loaded! Can detect {len(class_names)} diseases!")

# ── Prediction function ──────────────────────────────────────────────────────
def predict_disease(image):
    if image is None:
        return {}

    # Resize and normalize
    img = image.resize((224, 224))
    img_array = np.array(img)

    # Handle RGBA images (4 channels → 3 channels)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array, verbose=0)

    # Get top 3 results  ← FIX: removed * 100 (Gradio handles % internally)
    top3_indices = np.argsort(predictions[0])[-3:][::-1]

    results = {}
    for idx in top3_indices:
        disease = class_names[idx].replace("___", " → ").replace("_", " ")
        confidence = float(predictions[0][idx])   # ← BUG FIXED HERE
        results[disease] = confidence

    return results

# ── Custom CSS for beautiful UI ──────────────────────────────────────────────
custom_css = """
    /* Page background */
    .gradio-container {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364) !important;
        min-height: 100vh;
    }

    /* Title */
    h1 {
        font-size: 2.4rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }

    /* Description */
    .description {
        color: #a8d5a2 !important;
        text-align: center;
        font-size: 1rem !important;
    }

    /* Cards / panels */
    .block {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(10px);
    }

    /* Labels inside panels */
    label span {
        color: #a8d5a2 !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.05em;
    }

    /* Submit button */
    button.primary {
        background: linear-gradient(90deg, #56ab2f, #a8e063) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #0f2027 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 12px 0 !important;
        transition: opacity 0.2s;
    }
    button.primary:hover { opacity: 0.88; }

    /* Clear button */
    button.secondary {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Prediction label bars */
    .label-container {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        margin-bottom: 6px !important;
    }

    /* Footer */
    footer { display: none !important; }
"""

# ── Banner / description HTML ────────────────────────────────────────────────
banner = """
<div style="text-align:center; padding: 10px 0 18px;">
    <p style="color:#a8d5a2; font-size:1rem; margin:0;">
        🧠 Powered by <b>MobileNetV2</b> Transfer Learning &nbsp;|&nbsp;
        🌱 Trained on <b>54,305</b> leaf images &nbsp;|&nbsp;
        🎯 <b>94.16%</b> Validation Accuracy
    </p>
    <p style="color:#7fb3a0; font-size:0.85rem; margin-top:6px;">
        Upload a plant leaf photo — your AI will detect the disease instantly!
    </p>
</div>
"""

footer_html = """
<div style="text-align:center; margin-top:20px; color:#7fb3a0; font-size:0.82rem;">
    Built by <b style="color:#a8d5a2;">Bramha Vinayak Gulavani</b> &nbsp;·&nbsp;
    AI & ML Student, VIT Pune &nbsp;·&nbsp;
    <a href="https://github.com/bramhagulavani/plant-disease-ai"
       style="color:#56ab2f;" target="_blank">GitHub →</a>
</div>
"""

# ── Build Gradio app ─────────────────────────────────────────────────────────
with gr.Blocks(css=custom_css, title="🌿 Plant Disease Detector") as app:

    gr.HTML("<h1>🌿 Plant Disease Detector</h1>")
    gr.HTML(banner)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="📷 Upload a Leaf Photo",
                height=320
            )
            with gr.Row():
                clear_btn  = gr.ClearButton(
                    components=[image_input],
                    value="🗑️ Clear"
                )
                submit_btn = gr.Button("🔍 Detect Disease", variant="primary")

        with gr.Column(scale=1):
            output_label = gr.Label(
                num_top_classes=3,
                label="🌡️ Disease Prediction"
            )
            gr.HTML("""
                <div style="margin-top:14px; padding:14px 16px;
                            background:rgba(255,255,255,0.05);
                            border:1px solid rgba(255,255,255,0.1);
                            border-radius:12px; color:#a8d5a2;
                            font-size:0.84rem; line-height:1.7;">
                    <b style="color:#fff;">ℹ️ How to use:</b><br>
                    1. Upload a clear photo of a plant leaf<br>
                    2. Click <b>Detect Disease</b><br>
                    3. See top 3 predictions with confidence %<br><br>
                    <b style="color:#fff;">🌿 Supported plants:</b><br>
                    Apple, Tomato, Potato, Corn, Grape,
                    Strawberry, Peach, Pepper, Cherry & more
                </div>
            """)

    submit_btn.click(
        fn=predict_disease,
        inputs=image_input,
        outputs=output_label
    )

    gr.HTML(footer_html)

# ── Launch ───────────────────────────────────────────────────────────────────
print("Starting app...")
app.launch()