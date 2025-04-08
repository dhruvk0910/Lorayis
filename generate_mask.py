import cv2
import numpy as np
import os
from config import DATASET_ROOT, SELECTED_CLASS, IMAGE_SIZE

# --- Paths ---
test_dir = os.path.join(DATASET_ROOT, SELECTED_CLASS, "test", "good")
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
original_path = os.path.join(test_dir, test_images[0])

heatmap_path = f"outputs/heatmap_{SELECTED_CLASS}.png"
save_mask_path = f"outputs/mask_overlay_{SELECTED_CLASS}.png"

# --- Load images ---
original = cv2.imread(original_path)
heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)

# --- Resize heatmap to match original ---
heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))

# --- Threshold the heatmap to generate binary mask ---
_, mask = cv2.threshold(heatmap_resized, 127, 255, cv2.THRESH_BINARY)

# --- Convert single-channel mask to RGB ---
mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

# --- Apply red color to mask ---
mask_colored[np.where((mask_colored == [255, 255, 255]).all(axis=2))] = [0, 0, 255]

# --- Overlay mask on original ---
overlay = cv2.addWeighted(original, 0.7, mask_colored, 0.3, 0)

# --- Save result ---
os.makedirs("outputs", exist_ok=True)
cv2.imwrite(save_mask_path, overlay)

print(f"✅ Anomaly mask overlay saved to {save_mask_path}")
