from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from model import ConvAutoencoder

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# --- Load model ---
model = ConvAutoencoder()
model.load_state_dict(torch.load("saved_models/autoencoder_bottle.pth", map_location=torch.device('cpu')))
model.eval()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def process_image(image_path):
    original = Image.open(image_path).convert("RGB")
    original_tensor = transform(original).unsqueeze(0)

    with torch.no_grad():
        reconstructed = model(original_tensor)

    original_np = original_tensor.squeeze().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed.squeeze().permute(1, 2, 0).numpy()

    error_map = np.abs(original_np - reconstructed_np)
    error_map_gray = np.mean(error_map, axis=2)
    norm_map = (error_map_gray - error_map_gray.min()) / (error_map_gray.max() - error_map_gray.min() + 1e-8)
    heatmap_uint8 = (norm_map * 255).astype(np.uint8)

    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_cv = (original_np * 255).astype(np.uint8)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(original_cv, 0.6, heatmap_colored, 0.4, 0)

    # Save results
    heatmap_path = os.path.join(RESULT_FOLDER, "heatmap.png")
    overlay_path = os.path.join(RESULT_FOLDER, "overlay.png")
    cv2.imwrite(heatmap_path, heatmap_colored)
    cv2.imwrite(overlay_path, overlay)

    return heatmap_path, overlay_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)
            heatmap, overlay = process_image(img_path)

            return render_template("index.html", 
                                   uploaded_image=file.filename, 
                                   heatmap=os.path.basename(heatmap), 
                                   overlay=os.path.basename(overlay))

    return render_template("index.html")

@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/static/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
