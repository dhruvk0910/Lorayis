import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from model import ConvAutoencoder
from config import DATASET_ROOT, SELECTED_CLASS, IMAGE_SIZE, MODEL_SAVE_PATH

# --- Paths ---
test_dir = os.path.join(DATASET_ROOT, SELECTED_CLASS, "test", "good")
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
original_path = os.path.join(test_dir, test_images[0])
save_path = f"outputs/overlay_{SELECTED_CLASS}.png"

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# --- Load image ---
original = Image.open(original_path).convert("RGB")
original_tensor = transform(original).unsqueeze(0)

# --- Load model ---
model = ConvAutoencoder()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device('cpu')))
model.eval()

# --- Reconstruct ---
with torch.no_grad():
    reconstructed = model(original_tensor)

# --- Convert tensors to numpy ---
original_np = original_tensor.squeeze().permute(1, 2, 0).numpy()
reconstructed_np = reconstructed.squeeze().permute(1, 2, 0).numpy()

# --- Compute pixel-wise error map ---
error_map = np.abs(original_np - reconstructed_np)
error_map_gray = np.mean(error_map, axis=2)

# --- Normalize error map ---
error_map_norm = (error_map_gray - error_map_gray.min()) / (error_map_gray.max() - error_map_gray.min() + 1e-8)
error_map_uint8 = (error_map_norm * 255).astype(np.uint8)

# --- Create overlay ---
heatmap_color = cv2.applyColorMap(error_map_uint8, cv2.COLORMAP_JET)
original_cv = (original_np * 255).astype(np.uint8)
original_cv = cv2.cvtColor(original_cv, cv2.COLOR_RGB2BGR)
overlayed = cv2.addWeighted(original_cv, 0.6, heatmap_color, 0.4, 0)

# --- Save and Show ---
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(save_path, overlayed)
print(f"🔥 Overlay saved to {save_path}")

# Optional: Show in window
cv2.imshow("Overlay", overlayed)
cv2.waitKey(0)
cv2.destroyAllWindows()

