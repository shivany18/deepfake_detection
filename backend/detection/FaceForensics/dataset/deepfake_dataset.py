import os
from PIL import Image
from torch.utils.data import Dataset

class DeepfakeDataset(Dataset):
    def _init_(self, real_dir, fake_dir, transform=None, limit=None):
        self.transform = transform
        self.images = []
        self.labels = []

        real_images = os.listdir(real_dir)
        fake_images = os.listdir(fake_dir)

        if limit:
            real_images = real_images[:limit // 2]
            fake_images = fake_images[:limit // 2]

        for img in real_images:
            self.images.append(os.path.join(real_dir, img))
            self.labels.append(0)

        for img in fake_images:
            self.images.append(os.path.join(fake_dir, img))
            self.labels.append(1)

    def _len_(self):
        return len(self.images)

    def _getitem_(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label