import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MVTecDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        for img_name in os.listdir(root_dir):
            if img_name.endswith('.png') or img_name.endswith('.jpg'):
                self.image_paths.append(os.path.join(root_dir, img_name))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image
