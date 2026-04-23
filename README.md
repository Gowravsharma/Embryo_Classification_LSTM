# Embryo Development Phase Classification (CNN + LSTM)

##  Overview

This project focuses on **predicting embryo developmental phases** from time-lapse image sequences using a combination of:

* **CNN (MobileNetV2)** → spatial feature extraction
* **LSTM** → temporal sequence modeling
* **Ordinal-aware loss** → respects developmental ordering of phases

The model learns both **visual patterns** and **temporal progression**, which is critical for embryo development analysis.

---

##  Model Architecture

```
Image Sequence (B, T, C, H, W)
        │
        ▼
MobileNetV2 (partially fine-tuned)
        │
        ▼
Feature Sequence (B, T, 1280)
        │
        ▼
LSTM (temporal modeling)
        │
        ▼
Classifier (FC Layer)
        │
        ▼
Phase Prediction
```

---

##  Key Features

* Partial fine-tuning of MobileNetV2 (last layers unfrozen)
* Sequence modeling with LSTM
* Custom **Ordinal Distance Loss**
* Handles **class imbalance** using weighted CE + sampler
* Video-wise train/val/test split (no data leakage)
* Early stopping for stable training

---

## Dataset Structure

Each sample consists of:

* A sequence of frames from a video
* Corresponding phase label (last frame in sequence)

Example:

```
video_01/
    frame_001.jpg
    frame_002.jpg
    ...
```

---

## Data Pipeline

1. Frames grouped by video
2. Sorted temporally
3. Sliding window sequences created
4. Video-wise split into train/val/test
5. Loaded as:

```
(Batch, Sequence Length, Channels, Height, Width)
```

---

## Loss Function

Total loss:

```
L_total = L_weighted_CE + λ * L_ordinal
```

### Components:

* **Weighted Cross Entropy** → handles imbalance
* **Ordinal Distance Loss** → penalizes phase distance

---

## Training

### Recommended Settings

```python
BATCH_SIZE = 4–8
SEQ_LEN = 8–12
LR = 1e-3 (LSTM), 1e-5 (CNN)
```

### Run Training

```bash
python train.py
```

---

## Training Time

Approximate (on T4 GPU):

| Batch Size | Time/Epoch |
| ---------- | ---------- |
| 4          | 10–15 min  |
| 8          | 6–10 min   |

---

## Evaluation Metrics

* Accuracy
* Loss (CE + Ordinal)
* Confusion Matrix (optional)

---

## Important Notes

* Do **NOT** precompute features if training CNN end-to-end
* Ensure dataset returns `(B, T, C, H, W)`
* Use different learning rates for CNN and LSTM
* Reduce batch size if CUDA OOM occurs

---

##  Dependencies

* Python 3.10+
* PyTorch
* torchvision
* numpy
* PIL

Install:

```bash
pip install torch torchvision numpy pillow
```

---

##  Future Improvements

* Transformer-based temporal modeling
* Attention over frames
* Multi-scale feature extraction
* Self-supervised pretraining

---

##  License

This project is for research and educational purposes.

---

##  Author

Gowrav Sharma
