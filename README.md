# Deepfake Video Detection

A deep learning model that detects AI-generated (deepfake) videos using **EfficientNet-B0**, trained on the FaceForensics++ dataset. Achieves **~83% accuracy** and **0.918 AUC-ROC** on the test set.

---

## Overview

With the rapid rise of synthetic media, distinguishing real videos from AI-manipulated ones has become a critical problem. This project tackles binary video classification — real vs. deepfake — by extracting face frames from videos and running them through a fine-tuned EfficientNet-B0 convolutional neural network.

---

## Dataset

**[FaceForensics++](https://github.com/ondyari/FaceForensics)**

- Contains real and manipulated face videos generated using methods like Deepfakes, Face2Face, FaceSwap, and NeuralTextures
- Frames are extracted from videos and fed into the model as images
- Preprocessing includes face cropping, resizing to 224×224, and normalization

---

## Model Architecture

- **Base model:** EfficientNet-B0 (pretrained on ImageNet)
- **Approach:** Transfer learning — frozen base layers with a custom classification head
- **Output:** Binary classification (Real / Fake)
- **Framework:** PyTorch
- **Training environment:** Google Colab (GPU)

---

## Results

| Metric | Value |
|---|---|
| Accuracy | ~83% |
| AUC-ROC | 0.918 |

The high AUC-ROC indicates strong discriminative ability between real and fake faces even where raw accuracy has room to grow.

---

## Project Structure

```
deepfake-detection/
├── data/
│   └── (FaceForensics++ frames — not included, see dataset section)
├── notebooks/
│   └── deepfake_detection.ipynb   # Full training pipeline
├── model/
│   └── efficientnet_deepfake.pth  # Saved model weights
├── src/
│   ├── dataset.py                 # Dataset class and preprocessing
│   ├── model.py                   # Model definition
│   └── train.py                   # Training loop
└── README.md
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/KishoreGanesh05/deepfake-detection.git
cd deepfake-detection
```

### 2. Install dependencies
```bash
pip install torch torchvision efficientnet-pytorch opencv-python scikit-learn matplotlib
```

### 3. Prepare the dataset
Download FaceForensics++ frames and place them under `data/` with the structure:
```
data/
├── real/
└── fake/
```

### 4. Train the model
```bash
python src/train.py
```

Or run the full notebook end-to-end:
```
notebooks/deepfake_detection.ipynb
```

---

## Future Improvements

- [ ] **Deploy via Gradio on Hugging Face Spaces** — upload a video clip and get a real/fake prediction in the browser
- [ ] **Improve accuracy beyond 83%** — experiment with EfficientNet-B3/B4, data augmentation, and longer training
- [ ] **Video-level aggregation** — currently frame-level; aggregate predictions across frames for more robust video-level classification
- [ ] **Multi-method generalization** — test against newer deepfake generation techniques beyond FaceForensics++ methods
- [ ] **Explainability** — add Grad-CAM visualizations to highlight which facial regions the model focuses on

---

## Tech Stack

- Python, PyTorch
- EfficientNet-B0 (via `efficientnet-pytorch`)
- OpenCV (frame extraction)
- Scikit-learn (evaluation metrics)
- Google Colab

---

## References

- [FaceForensics++ Paper](https://arxiv.org/abs/1901.08971)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

---

*Built by [Kishore Ganesh](https://github.com/KishoreGanesh05)*
