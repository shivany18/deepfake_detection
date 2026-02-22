import torch
import timm
from PIL import Image
from torchvision import transforms
import io

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("xception", pretrained=False, num_classes=2)
model.load_state_dict(torch.load("models/xception_deepfake.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def detect_image(file):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence_tensor, pred = torch.max(probs, 1)

    confidence = confidence_tensor.item() * 100
    confidence = max(0.0, min(confidence, 100.0))

    label = "FAKE" if pred.item() == 1 else "REAL"

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }