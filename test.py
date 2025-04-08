import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import ConvAutoencoder
import os
from config import DATASET_ROOT, SELECTED_CLASS, MODEL_SAVE_PATH, IMAGE_SIZE

# --- Load model ---
model = ConvAutoencoder()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=torch.device("cpu")))
model.eval()

# --- Pick a test image ---
test_image_dir = os.path.join(DATASET_ROOT, SELECTED_CLASS, "test", "good")
test_image_list = [f for f in os.listdir(test_image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
img_path = os.path.join(test_image_dir, test_image_list[0])  # Just pick the first image

# --- Load and transform image ---
image = Image.open(img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor()
])
input_tensor = transform(image).unsqueeze(0)

# --- Reconstruct ---
with torch.no_grad():
    reconstructed = model(input_tensor)

# --- Convert to image ---
reconstructed_img = reconstructed.squeeze(0).permute(1, 2, 0).numpy()
reconstructed_img = (reconstructed_img * 255).astype("uint8")
reconstructed_pil = Image.fromarray(reconstructed_img)

# --- Save ---
save_path = f"reconstructed_{SELECTED_CLASS}.png"
reconstructed_pil.save(save_path)
print(f"✅ Reconstructed image saved at: {save_path}")

# --- Plot ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_pil)
plt.title("Reconstructed Image")
plt.axis("off")

plt.tight_layout()
plt.show()
