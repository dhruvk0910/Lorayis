import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MVTecDataset
from model import ConvAutoencoder
import os
from config import SELECTED_CLASS, DATASET_ROOT, IMAGE_SIZE, MODEL_SAVE_PATH

# --- Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# --- Dataset ---
train_data_path = os.path.join(DATASET_ROOT, SELECTED_CLASS, "train", "good")
train_dataset = MVTecDataset(train_data_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Model ---
model = ConvAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training Loop ---
print(f"🚀 Starting training for class: {SELECTED_CLASS}")
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {epoch_loss:.6f}")

# --- Save Model ---
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Training complete. Model saved to: {MODEL_SAVE_PATH}")

