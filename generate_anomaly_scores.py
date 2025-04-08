import os
import csv
import torch
import numpy as np
from PIL import Image
from model import ConvAutoencoder
import torchvision.transforms as transforms

# Paths
test_dir = "C:/Users/amish/Lorayis/data/test"
model_path = "C:/Users/amish/Lorayis/saved_models/autoencoder_bottle.pth"
csv_path = "C:/Users/amish/Lorayis/outputs/anomaly_scores.csv"

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

# Process images
rows = [("filename", "anomaly_score_mean", "anomaly_score_max")]

for filename in os.listdir(test_dir):
    if not filename.endswith(".png"):
        continue

    img_path = os.path.join(test_dir, filename)
    image = Image.open(img_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstructed = model(image_tensor)

    original_np = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze().permute(1, 2, 0).cpu().numpy()

    error_map = np.abs(original_np - reconstructed_np)
    error_map_gray = np.mean(error_map, axis=2)
    error_map_gray = (error_map_gray - error_map_gray.min()) / (error_map_gray.max() - error_map_gray.min() + 1e-8)

    anomaly_mean = float(np.mean(error_map_gray))
    anomaly_max = float(np.max(error_map_gray))

    rows.append((filename, anomaly_mean, anomaly_max))
    print(f"[✓] Scored {filename} - Mean: {anomaly_mean:.4f}, Max: {anomaly_max:.4f}")

# Save to CSV
os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # ← THIS LINE FIXES IT

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"\n📊 Anomaly scores saved to {csv_path}")
