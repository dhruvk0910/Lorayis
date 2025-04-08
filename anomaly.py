import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from model import ConvAutoencoder
from config import DATASET_ROOT, SELECTED_CLASS, IMAGE_SIZE, MODEL_SAVE_PATH

# --- Build paths ---
test_dir = os.path.join(DATASET_ROOT, SELECTED_CLASS, "test", "good")
test_image_list = [f for f in os.listdir(test_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
original_path = os.path.join(test_dir, test_image_list[0])
save_heatmap_path = f"outputs/heatmap_{SELECTED_CLASS}.png"

# --- Image transforms ---
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

# --- Compute pixel-wise absolute difference (error map) ---
error_map = np.abs(original_np - reconstructed_np)
error_map_gray = np.mean(error_map, axis=2)

# --- Normalize error map to [0, 1] ---
error_map_gray = (error_map_gray - error_map_gray.min()) / (error_map_gray.max() - error_map_gray.min() + 1e-8)

# --- Save heatmap ---
os.makedirs("outputs", exist_ok=True)
plt.imsave(save_heatmap_path, error_map_gray, cmap='hot')
print(f"🔥 Heatmap saved to {save_heatmap_path}")

# --- Plot ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].imshow(original_np)
axs[0].set_title("Original Image")
axs[0].axis('off')

axs[1].imshow(reconstructed_np)
axs[1].set_title("Reconstructed Image")
axs[1].axis('off')

axs[2].imshow(error_map_gray, cmap='hot')
axs[2].set_title("Anomaly Heatmap")
axs[2].axis('off')

plt.tight_layout()
plt.show()
