"""Small web interface for the CIFAR-10 model in train.py."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "cifar_net.pth"
CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


class Net(nn.Module):
    """The same network layout used by train.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout1 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 2 * 2)
        x = self.dropout1(F.relu(self.fc1(x)))
        return self.fc2(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
load_error = None
if WEIGHTS_PATH.exists():
    try:
        model = Net().to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        model.eval()
    except (RuntimeError, OSError, EOFError) as exc:
        model = None
        load_error = str(exc)

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024


@app.get("/")
def home():
    return render_template("index.html", model_ready=model is not None)


@app.post("/predict")
def predict():
    if model is None:
        message = "Model weights are missing. Run train.py first to create cifar_net.pth."
        if load_error:
            message = "Model weights could not be loaded. Run train.py again to create a fresh cifar_net.pth."
        return jsonify({"error": message}), 503
    if "image" not in request.files:
        return jsonify({"error": "Choose an image to classify."}), 400
    uploaded = request.files["image"]
    if not uploaded.filename:
        return jsonify({"error": "Choose an image to classify."}), 400
    try:
        image = Image.open(uploaded.stream).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify({"error": "That file could not be read as an image."}), 400
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        values, indices = torch.topk(probabilities, k=3)
    predictions = [
        {"class": CLASSES[index], "confidence": round(value * 100, 1)}
        for value, index in zip(values.tolist(), indices.tolist())
    ]
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
