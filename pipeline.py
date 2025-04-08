import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model import ConvAutoencoder
from config import DATASET_ROOT, SELECTED_CLASS, IMAGE_SIZE

# --- Paths ---
test_dir = os.path.join(DATASET_ROOT, SELECTED_CLASS, "test", "good")
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
image_path = os.path.join(test_dir, test_images[0])

model_path = f"saved_models/autoencoder_{SELECTED_CLASS}.pth"
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# --- Load model ---
def load_model():
    model = ConvAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# --- Step 1: Load and preprocess image ---
def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

# --- Step 2: Reconstruct image ---
def reconstruct_image(model, input_tensor):
    with torch.no_grad():
        return model(input_tensor)

# --- Step 3: Save reconstructed image ---
def save_reconstructed(original_tensor, reconstructed_tensor):
    rec_np = reconstructed_tensor.squeeze().permute(1, 2, 0).numpy()
    rec_np = (rec_np * 255).astype(np.uint8)
    Image.fromarray(rec_np).save(f"{outputs_dir}/reconstructed_{SELECTED_CLASS}.png")
    print("✅ Reconstructed image saved.")

# --- Step 4: Generate and save heatmap ---
def generate_heatmap(original_tensor, reconstructed_tensor):
    orig_np = original_tensor.squeeze().permute(1, 2, 0).numpy()
    rec_np = reconstructed_tensor.squeeze().permute(1, 2, 0).numpy()
    error = np.abs(orig_np - rec_np)
    error_gray = np.mean(error, axis=2)
    error_norm = (error_gray - error_gray.min()) / (error_gray.max() - error_gray.min() + 1e-8)
    plt.imsave(f"{outputs_dir}/heatmap_{SELECTED_CLASS}.png", error_norm, cmap='hot')
    print("🔥 Heatmap saved.")
    return error_norm, orig_np

# --- Step 5: Overlay heatmap ---
def overlay_heatmap(orig_np, error_norm):
    heatmap_uint8 = (error_norm * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_cv = (orig_np * 255).astype(np.uint8)
    original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)
    cv2.imwrite(f"{outputs_dir}/heatmap_overlay_{SELECTED_CLASS}.png", overlay)
    print("🖼️ Heatmap overlay saved.")

# --- Step 6: Generate binary anomaly mask ---
def generate_anomaly_mask_overlay():
    original = cv2.imread(image_path)
    heatmap_gray = cv2.imread(f"{outputs_dir}/heatmap_{SELECTED_CLASS}.png", cv2.IMREAD_GRAYSCALE)
    heatmap_resized = cv2.resize(heatmap_gray, (original.shape[1], original.shape[0]))
    _, mask = cv2.threshold(heatmap_resized, 127, 255, cv2.THRESH_BINARY)
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]
    overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite(f"{outputs_dir}/mask_overlay_{SELECTED_CLASS}.png", overlay)
    print("🎯 Anomaly mask overlay saved.")

# --- Run full pipeline ---
def run_pipeline():
    print(f"\n🚀 Running pipeline for class: {SELECTED_CLASS}\n")
    input_tensor = load_image(image_path)
    model = load_model()
    reconstructed_tensor = reconstruct_image(model, input_tensor)
    save_reconstructed(input_tensor, reconstructed_tensor)
    error_map, orig_np = generate_heatmap(input_tensor, reconstructed_tensor)
    overlay_heatmap(orig_np, error_map)
    generate_anomaly_mask_overlay()
    print("\n✅ Full pipeline complete.\n")

if __name__ == "__main__":
    run_pipeline()
