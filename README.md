# IML Project

The code covers data loading, exploration, preprocessing, augmentation, model training, evaluation, and visualization.

---

## 1. Project Structure

```
project/
│
├── dataset/
│   ├── allowed/        # Class 1 audio (.wav)
│   └── not_allowed/    # Class 0 audio (.wav)
│
├── models/             # Saved model checkpoints
├── outputs/            # Training curves (loss & F1)
├── notebook.py / .ipynb
└── README.md
```

---

## 2. Requirements

Python libraries used:
- numpy
- matplotlib
- librosa
- soundfile
- scikit‑learn
- torch

Make sure PyTorch and librosa are installed before running.

---

## 3. Configuration Parameters

All global configuration is defined in the **Args class** near the top of the file:

```python
args.data = "./dataset"      # Dataset root folder
args.sr = 16000               # Audio sample rate
args.n_mels = 64              # Number of Mel bands
args.fixed_length = 200       # Time frames per spectrogram

args.batch_size = 32
args.lr = 1e-3                # Learning rate
args.weight_decay = 1e-4      # L2 regularization

args.test_size = 0.15         # Test split
args.val_size = 0.15          # Validation split
```

These parameters control **data resolution, training speed, and regularization**.

---

## 4. Dataset Handling

### File Discovery

Audio files are collected using:
```python
build_filelist(root)
```
Expected folder names:
- `allowed` → label 1
- `not_allowed` → label 0

### Class Balancing

The dataset is **balanced manually** by down‑sampling the larger class:
```python
allowed = random.sample(allowed, min_len)
not_allowed = random.sample(not_allowed, min_len)
```

---

## 5. Exploratory Data Analysis (EDA)

A histogram of audio durations is plotted using:
```python
librosa.get_duration()
```
This helps justify the choice of `fixed_length`.

---

## 6. Audio Preprocessing Pipeline

### 6.1 Audio Loading

```python
load_audio(path, sr)
```
- Resamples audio to `args.sr`

### 6.2 Spectrogram Creation

```python
make_mel(y, sr)
```
Inside this function:
- Mel spectrogram is computed
- **Log scaling (normalization #1)** is applied:
```python
librosa.power_to_db(S, ref=np.max)
```

### 6.3 Statistical Normalization (Normalization #2)

Inside `SpectrogramDataset.__getitem__`:
```python
S = (S - S.mean()) / (S.std() + 1e-9)
```
This enforces **zero mean and unit variance per sample**.

---

## 7. Data Augmentation (Overfitting Control)

Enabled only for **training data**:

```python
SpectrogramDataset(train_pairs, augment=True)
```

Augmentations include:
- Additive noise (`add_noise`)
- Pitch shifting (`pitch_shift`)
- Time stretching (`time_stretch`)

These are defined inside the `SpectrogramDataset` class.

---

## 8. Train / Validation / Test Split

Performed using **stratified splitting**:
```python
train_test_split(..., stratify=labels)
```

Resulting loaders:
- `train_loader` (augmented)
- `val_loader` (clean)
- `test_loader` (clean, acts as "real test")

---

## 9. Model Architecture

Defined in `SimpleConvNet`:

```python
Conv2d → ReLU → MaxPool
Conv2d → ReLU → MaxPool
Dropout(0.3)
Fully Connected → 2 classes
```

### Dropout Location

```python
self.dropout = nn.Dropout(0.3)
```
Applied **after convolution layers**, before the classifier.

---

## 10. Optimization & Loss

Defined at the start of the training cell:

```python
criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay
)
```

- **Adam optimizer**
- **Weight decay** = L2 regularization

---

## 11. Training Loop

- Runs in an **infinite loop** (Ctrl+C to stop)
- Tracks:
  - Loss
  - Train F1
  - Validation F1
  - Real Test F1

Metrics are computed using:
```python
f1_score(..., average="macro")
```

---

## 12. Evaluation

Evaluation logic is separated in:
```python
evaluate(model, loader)
```

Used for validation and test sets.

---

## 13. Outputs & Checkpoints

### Model Saving

Each epoch:
```python
torch.save(model.state_dict(), "./models/model_epoch_X.pth")
```

### Training Curves

Saved to `./outputs/`:
- Loss
- Train F1
- Validation F1
- Real Test F1

Plotted automatically per epoch.

---

## 14. How to Run

1. Place dataset in `./dataset/allowed` and `./dataset/not_allowed`
2. Run the notebook / script from top to bottom
3. Stop training manually when convergence is reached (Ctrl+C)

---
