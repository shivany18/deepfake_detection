def main():
    import os
    import torch
    import timm
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    from torchvision import transforms
    from PIL import Image
    from tqdm import tqdm

    # ---------------- PATHS ----------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_DIR = os.path.join(BASE_DIR, "..", "backend", "140k_faces", "real_vs_fake", "real-vs-fake", "train")
    REAL_DIR = os.path.join(DATASET_DIR, "real")
    FAKE_DIR = os.path.join(DATASET_DIR, "fake")

    MODEL_DIR = os.path.join(BASE_DIR, "..", "backend", "models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------------- DEVICE ----------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)

    # ---------------- CPU OPTIMIZATION ----------------
    torch.set_num_threads(8)
    torch.backends.mkldnn.enabled = True

    # ---------------- DATASET ----------------
    class DeepfakeDataset(Dataset):
        def __init__(self, real_dir, fake_dir, transform=None, limit=None):
            self.data = []
            self.transform = transform

            real_images = os.listdir(real_dir)
            fake_images = os.listdir(fake_dir)

            if limit:
                real_images = real_images[:limit // 2]
                fake_images = fake_images[:limit // 2]

            for img in real_images:
                self.data.append((os.path.join(real_dir, img), 0))  # REAL

            for img in fake_images:
                self.data.append((os.path.join(fake_dir, img), 1))  # FAKE

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            path, label = self.data[idx]
            image = Image.open(path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            return image, label

    # ---------------- TRANSFORMS ----------------
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # ---------------- LOAD DATA ----------------
    full_dataset = DeepfakeDataset(REAL_DIR, FAKE_DIR, transform=train_transform, limit=20000)
    print("Total images:", len(full_dataset))

    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,      # WINDOWS SAFE
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,      # WINDOWS SAFE
        pin_memory=False
    )

    # ---------------- MODEL ----------------
    model = timm.create_model("xception", pretrained=True, num_classes=2)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier
    for param in model.get_classifier().parameters():
        param.requires_grad = True

    model.to(DEVICE)

    # ---------------- RESUME CHECKPOINT ----------------
    checkpoint_path = os.path.join(MODEL_DIR, "xception_deepfake.pth")

    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        print("✅ Loaded previous checkpoint:", checkpoint_path)
    else:
        print("ℹ️ No previous checkpoint found. Training from scratch.")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4
    )

    # ---------------- TRAINING CONFIG ----------------
    EPOCHS_PHASE1 = 5
    EPOCHS_PHASE2 = 8
    best_val_acc = 0.0

    # ---------------- TRAIN FUNCTIONS ----------------
    def train_one_epoch(model, loader, optimizer, criterion):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return running_loss / len(loader), correct / total

    def validate(model, loader, criterion):
        model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return running_loss / len(loader), correct / total

    # ---------------- PHASE 1 ----------------
    print("\n🔥 Phase 1: Training classifier head\n")

    for epoch in range(EPOCHS_PHASE1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"[Phase 1][Epoch {epoch+1}/{EPOCHS_PHASE1}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "xception_best.pth"))
            print("✅ Best model saved!")

        # Always update resume checkpoint
        torch.save(model.state_dict(), checkpoint_path)

    # ---------------- UNFREEZE TOP LAYERS ----------------
    print("\n🔓 Unfreezing top Xception layers...\n")

    for name, param in model.named_parameters():
        if "blocks.12" in name or "blocks.13" in name:
            param.requires_grad = True

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,
        weight_decay=1e-4
    )

    # ---------------- PHASE 2 ----------------
    print("\n🔥 Phase 2: Fine-tuning\n")

    for epoch in range(EPOCHS_PHASE2):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate(model, val_loader, criterion)

        print(f"[Phase 2][Epoch {epoch+1}/{EPOCHS_PHASE2}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "xception_best.pth"))
            print("✅ Best model saved!")

        # Always update resume checkpoint
        torch.save(model.state_dict(), checkpoint_path)

    # ---------------- FINAL SAVE ----------------
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "xception_final.pth"))
    print("\n🎉 Training complete.")
    print("Best Validation Accuracy:", best_val_acc)


if __name__ == "__main__":
    main()