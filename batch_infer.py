import os
import cv2
import torch
import numpy as np
from PIL import Image
from model import ConvAutoencoder
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Paths
test_dir = "C:/Users/amish/Lorayis/data/test"
model_path = "C:/Users/amish/Lorayis/saved_models/autoencoder_bottle.pth"
output_dir = "C:/Users/amish/Lorayis/outputs"

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Make sure output dir exists
os.makedirs(output_dir, exist_ok=True)

# Loop through test images
for filename in os.listdir(test_dir):
    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(test_dir, filename)
    output_path = os.path.join(output_dir, filename.split('.')[0])
    os.makedirs(output_path, exist_ok=True)

    # Load image
    original = Image.open(img_path).convert("RGB")
    original_tensor = transform(original).unsqueeze(0).to(device)

    # Reconstruct
    with torch.no_grad():
        reconstructed = model(original_tensor)

    # Convert to numpy
    original_np = original_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()

    # Error map
    error_map = np.abs(original_np - reconstructed_np)
    error_map_gray = np.mean(error_map, axis=2)
    error_map_gray = (error_map_gray - error_map_gray.min()) / (error_map_gray.max() - error_map_gray.min() + 1e-8)

    # Save original & reconstruction
    Image.fromarray((original_np * 255).astype(np.uint8)).save(os.path.join(output_path, "original.png"))
    Image.fromarray((reconstructed_np * 255).astype(np.uint8)).save(os.path.join(output_path, "reconstruction.png"))

    # Save heatmap
    plt.imsave(os.path.join(output_path, "heatmap.png"), error_map_gray, cmap='hot')

    # Overlay heatmap
    heatmap_uint8 = (error_map_gray * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    original_cv = cv2.cvtColor((original_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_cv, 0.7, heatmap_color, 0.3, 0)
    cv2.imwrite(os.path.join(output_path, "overlay.png"), overlay)

    # Binary mask & overlay
    threshold = 0.4
    mask = (error_map_gray > threshold).astype(np.uint8) * 255
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    mask_overlay = cv2.addWeighted(original_cv, 0.7, mask_colored, 0.3, 0)
    cv2.imwrite(os.path.join(output_path, "mask_overlay.png"), mask_overlay)

    print(f"[✓] Processed {filename}")

print("\n🔥 All test images processed. Check your 'outputs/' folder.")
