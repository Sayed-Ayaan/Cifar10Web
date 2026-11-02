# Fieldnote — CIFAR-10 image lab

A lightweight Flask web app for classifying an uploaded image with the CNN defined in `train.py`.

## Run it

1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train.py` (this downloads CIFAR-10 and saves `cifar_net.pth`)
3. Start the website: `python app.py`
4. Open `http://127.0.0.1:5000`

Training runs for 50 epochs and may take a while. The web page can be opened before training to see the interface; predictions become available once `cifar_net.pth` exists.

The app resizes uploaded images to 32 × 32 and applies the same `Normalize((0.5,)*3, (0.5,)*3)` preprocessing as the training script. It reports the three most likely CIFAR-10 classes.
