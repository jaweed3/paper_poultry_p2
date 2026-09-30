# ML2 Paper Execution Instructions — Edge Poultry Disease Detection

> **Date:** August 15, 2026
> **Paper:** Quantization-Aware Deployment of Lightweight CNNs for Poultry Fecal Disease Detection on Resource-Constrained Edge Devices

---

## WORKING RULES

1. **Every command must be executed, not described.**
2. **Every output must be captured.** Save logs to `~/research/paper_poultry_edge/logs/`.
3. **Every result must be verified.** Check file exists, check metrics are reasonable.
4. **If a command fails, debug and fix.** Do not skip or assume success.
5. **Commit progress after each major step.** Git commit with descriptive messages.
6. **All paths are absolute.** Base directory: `~/research/paper_poultry_edge/`
7. **Python version:** Use Python 3.11 (brew). If not available, install: `brew install python@3.11`
8. **Virtual environment:** Always use venv. Never install globally.

---

## PHASE 0: Environment Setup

### 0.1 Create project structure

```bash
mkdir -p ~/research/paper_poultry_edge/{data,models/{fp32,ptq,qat,onnx},logs,figures,scripts,results}
cd ~/research/paper_poultry_edge
```

### 0.2 Initialize git

```bash
cd ~/research/paper_poultry_edge
git init
git branch -M main
```

### 0.3 Create `.gitignore`

```bash
cat > .gitignore << 'EOF'
data/
models/
logs/
__pycache__/
*.pyc
.venv/
*.egg-info/
wandb/
.ipynb_checkpoints/
EOF
```

### 0.4 Create virtual environment

```bash
cd ~/research/paper_poultry_edge
python3.11 -m venv .venv
source .venv/bin/activate
```

### 0.5 Install dependencies

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install onnx onnxruntime onnxruntime-tools
pip install numpy pandas scikit-learn matplotlib seaborn tqdm Pillow
pip install grad-cam  # for interpretability
pip install thop      # for FLOPs counting
```

### 0.6 Verify installation

```bash
python3 -c "
import torch; print(f'PyTorch: {torch.__version__}')
import torchvision; print(f'TorchVision: {torchvision.__version__}')
import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')
import numpy; print(f'NumPy: {numpy.__version__}')
print('All imports OK')
"
```

Save output to `logs/phase0_setup.txt`.

---

## PHASE 1: Dataset Preparation

### 1.1 Download dataset

```bash
cd ~/research/paper_poultry_edge
mkdir -p data/raw
cd data/raw

# Download from Zenodo (Machuve et al. 2021)
wget -O poultry_fecal.zip "https://zenodo.org/records/4628934/files/PoultryDiseases.zip?download=1"

# If wget fails, try:
# curl -L -o poultry_fecal.zip "https://zenodo.org/records/4628934/files/PoultryDiseases.zip?download=1"
```

### 1.2 Extract dataset

```bash
cd ~/research/paper_poultry_edge/data/raw
unzip poultry_fecal.zip
ls -la  # Verify structure: should have folders for each class
```

### 1.3 Verify dataset structure

```bash
cd ~/research/paper_poultry_edge
python3 << 'EOF'
import os
from collections import Counter

data_dir = "data/raw"
classes = []
for root, dirs, files in os.walk(data_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            classes.append(os.path.basename(root))

counter = Counter(classes)
print("Dataset structure:")
print(f"Total images: {sum(counter.values())}")
for cls, count in sorted(counter.items()):
    print(f"  {cls}: {count} images")
EOF
```

Expected output: ~6,812 images across 4 classes (healthy, cocci, ncd, salmo).

Save output to `logs/phase1_dataset_info.txt`.

### 1.4 Organize dataset into train/val/test split

```bash
cd ~/research/paper_poultry_edge
python3 << 'EOF'
import os
import shutil
import random
from collections import defaultdict

# Configuration
RAW_DIR = "data/raw"
SPLIT_DIR = "data/split"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42

random.seed(SEED)

# Find all images by class
class_images = defaultdict(list)
for root, dirs, files in os.walk(RAW_DIR):
    cls = os.path.basename(root)
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            class_images[cls].append(os.path.join(root, f))

print(f"Found {sum(len(v) for v in class_images.values())} images in {len(class_images)} classes")

# Create split directories
for split in SPLITS:
    for cls in class_images:
        os.makedirs(os.path.join(SPLIT_DIR, split, cls), exist_ok=True)

# Split and copy
for cls, images in class_images.items():
    random.shuffle(images)
    n = len(images)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["val"])

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, files in splits.items():
        for f in files:
            shutil.copy2(f, os.path.join(SPLIT_DIR, split, cls))

    print(f"{cls}: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

print("\nDataset split complete!")
EOF
```

Save output to `logs/phase1_split_info.txt`.

---

## PHASE 2: Training Pipeline

### 2.1 Create training script

Create file `scripts/train.py`:

```python
#!/usr/bin/env python3
"""
Training script for poultry fecal disease detection.
Trains MobileNetV2, ShuffleNetV2, EfficientNet-Lite0 on fecal image dataset.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    "data_dir": "data/split",
    "output_dir": "models",
    "image_size": 224,
    "batch_size": 32,
    "epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_model(name, num_classes=4, pretrained=True):
    """Load model architecture with optional ImageNet pretrained weights."""

    if name == "mobilenetv2":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    elif name == "shufflenetv2":
        model = models.shufflenet_v2_x1_0(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "efficientnet_b0":
        # EfficientNet-Lite0 via torchvision
        model = models.efficientnet_b0(pretrained=pretrained)
        # Modify classifier for 4 classes
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model


# ============================================================
# DATA LOADING
# ============================================================

def get_data_loaders(data_dir, image_size, batch_size, num_workers):
    """Create train/val/test data loaders with augmentation."""

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, "val"), eval_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), eval_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_names


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validate model. Returns loss, accuracy, per-class accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] = class_total.get(label, 0) + 1
                if predicted[i].eq(labels[i]):
                    class_correct[label] = class_correct.get(label, 0) + 1

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    class_acc = {k: class_correct.get(k, 0) / v for k, v in class_total.items()}

    return epoch_loss, epoch_acc, class_acc


def train_model(model_name, model, train_loader, val_loader, config):
    """Full training loop with early stopping."""
    device = config["device"]
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"],
                          weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    output_dir = Path(config["output_dir"]) / "fp32" / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device)
        val_loss, val_acc, class_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{config['epochs']}: "
              f"Train Loss={train_loss:.4f} Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"  → New best model saved (val_acc={val_acc:.4f})")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pth")

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete for {model_name}")
    print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {output_dir}")

    return best_val_acc, best_epoch, history


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="all",
                       choices=["mobilenetv2", "shufflenetv2", "efficientnet_b0", "all"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    # Update config
    CONFIG["epochs"] = args.epochs
    CONFIG["batch_size"] = args.batch_size
    CONFIG["lr"] = args.lr

    # Set seed
    torch.manual_seed(CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(CONFIG["seed"])

    # Load data
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        CONFIG["data_dir"], CONFIG["image_size"], CONFIG["batch_size"], CONFIG["num_workers"]
    )

    # Define models
    models_to_train = {
        "mobilenetv2": "mobilenetv2",
        "shufflenetv2": "shufflenetv2",
        "efficientnet_b0": "efficientnet_b0"
    }

    if args.model != "all":
        models_to_train = {args.model: models_to_train[args.model]}

    # Train each model
    results = {}
    for name, model_type in models_to_train.items():
        print(f"\n{'#'*60}")
        print(f"# Training: {name}")
        print(f"{'#'*60}")

        model = get_model(model_type, num_classes=len(class_names))
        best_acc, best_epoch, history = train_model(
            name, model, train_loader, val_loader, CONFIG
        )
        results[name] = {
            "best_val_acc": best_acc,
            "best_epoch": best_epoch
        }

    # Save overall results
    results_path = Path(CONFIG["output_dir"]) / "fp32" / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"Results saved to: {results_path}")
    for name, res in results.items():
        print(f"  {name}: val_acc={res['best_val_acc']:.4f} (epoch {res['best_epoch']})")


if __name__ == "__main__":
    main()
```

### 2.2 Run training for all models

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/train.py --model all --epochs 50 2>&1 | tee logs/phase2_training.log
```

### 2.3 Verify trained models

```bash
cd ~/research/paper_poultry_edge
python3 << 'EOF'
import os
import json

models_dir = "models/fp32"
for model_name in os.listdir(models_dir):
    model_path = os.path.join(models_dir, model_name)
    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        print(f"\n{model_name}:")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(model_path, f)) / (1024*1024)
            print(f"  {f}: {size:.2f} MB")

        # Check history
        history_path = os.path.join(model_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
            print(f"  Final val_acc: {history['val_acc'][-1]:.4f}")
            print(f"  Best val_acc: {max(history['val_acc']):.4f}")
EOF
```

Save output to `logs/phase2_model_verification.txt`.

---

## PHASE 3: ONNX Export

### 3.1 Create export script

Create file `scripts/export_onnx.py`:

```python
#!/usr/bin/env python3
"""
Export trained PyTorch models to ONNX format.
"""

import os
import sys
from pathlib import Path

import torch
import torch.onnx

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import get_model

def export_to_onnx(model_name, input_shape=(1, 3, 224, 224), output_dir="models/onnx"):
    """Export PyTorch model to ONNX format."""

    # Load model architecture
    model = get_model(model_name, num_classes=4, pretrained=False)

    # Load trained weights
    weights_path = f"models/fp32/{model_name}/best_model.pth"
    if not os.path.exists(weights_path):
        weights_path = f"models/fp32/{model_name}/final_model.pth"

    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_shape)

    # Export
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name}.onnx")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(output_path)
    ort_output = session.run(None, {"input": dummy_input.numpy()})

    print(f"Exported: {output_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Output shape: {ort_output[0].shape}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

    return output_path

if __name__ == "__main__":
    models = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
    for model_name in models:
        print(f"\nExporting {model_name}...")
        export_to_onnx(model_name)
```

### 3.2 Export all models

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/export_onnx.py 2>&1 | tee logs/phase3_onnx_export.log
```

### 3.3 Verify ONNX models

```bash
cd ~/research/paper_poultry_edge
python3 << 'EOF'
import onnxruntime as ort
import os

onnx_dir = "models/onnx"
for f in sorted(os.listdir(onnx_dir)):
    if f.endswith(".onnx"):
        path = os.path.join(onnx_dir, f)
        session = ort.InferenceSession(path)
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        size_mb = os.path.getsize(path) / (1024*1024)
        print(f"{f}:")
        print(f"  Input: {input_meta.name} {input_meta.shape}")
        print(f"  Output: {output_meta.name} {output_meta.shape}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Provider: {session.get_providers()}")
        print()
EOF
```

Save output to `logs/phase3_onnx_verification.txt`.

---

## PHASE 4: Post-Training Quantization (PTQ)

### 4.1 Create PTQ script

Create file `scripts/quantize_ptq.py`:

```python
#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) for ONNX models.
Converts FP32 ONNX models to INT8.
"""

import os
import json
import numpy as np
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
from onnxruntime.quantization.calibrateDataReader import CalibrationDataReader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

class CalibrationDataReaderImpl(CalibrationDataReader):
    """Custom calibration reader for PTQ."""

    def __init__(self, data_dir, image_size=224, num_samples=500):
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        dataset = datasets.ImageFolder(data_dir, transform)
        self.loader = DataLoader(dataset, batch_size=1, shuffle=False)
        self.num_samples = min(num_samples, len(dataset))
        self.current = 0

    def get_next(self):
        if self.current >= self.num_samples:
            return None
        images, _ = next(iter(self.loader))
        self.current += 1
        return {"input": images.numpy()}


def ptq_dynamic(onnx_path, output_path):
    """Dynamic range quantization (weights only)."""
    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    return output_path


def ptq_static(onnx_path, output_path, calibration_data_dir, num_calib=500):
    """Static quantization (weights + activations)."""
    reader = CalibrationDataReaderImpl(calibration_data_dir, num_samples=num_calib)
    quantize_static(
        model_input=onnx_path,
        model_output=output_path,
        calibration_data_reader=reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8
    )
    return output_path


def evaluate_onnx(onnx_path, data_dir, image_size=224, num_samples=None):
    """Evaluate ONNX model accuracy."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(data_dir, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    correct = 0
    total = 0
    for i, (images, labels) in enumerate(loader):
        if num_samples and i >= num_samples:
            break
        outputs = session.run(None, {"input": images.numpy()})
        pred = np.argmax(outputs[0], axis=1)
        correct += (pred == labels.numpy()).sum()
        total += labels.size(0)

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    models = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
    calib_dir = "data/split/val"
    test_dir = "data/split/test"
    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"PTQ for {model_name}")
        print(f"{'='*60}")

        fp32_path = f"models/onnx/{model_name}.onnx"
        ptq_dyn_path = f"models/ptq/{model_name}_ptq_dynamic.onnx"
        ptq_static_path = f"models/ptq/{model_name}_ptq_static.onnx"

        os.makedirs("models/ptq", exist_ok=True)

        # Dynamic PTQ
        print("Running dynamic PTQ...")
        ptq_dynamic(fp32_path, ptq_dyn_path)
        acc_dyn = evaluate_onnx(ptq_dyn_path, test_dir)
        size_dyn = os.path.getsize(ptq_dyn_path) / (1024*1024)
        print(f"  Dynamic PTQ: acc={acc_dyn:.4f}, size={size_dyn:.2f} MB")

        # Static PTQ
        print("Running static PTQ (with calibration)...")
        ptq_static(fp32_path, ptq_static_path, calib_dir)
        acc_static = evaluate_onnx(ptq_static_path, test_dir)
        size_static = os.path.getsize(ptq_static_path) / (1024*1024)
        print(f"  Static PTQ: acc={acc_static:.4f}, size={size_static:.2f} MB")

        # FP32 baseline
        acc_fp32 = evaluate_onnx(fp32_path, test_dir)
        size_fp32 = os.path.getsize(fp32_path) / (1024*1024)

        results[model_name] = {
            "fp32": {"accuracy": acc_fp32, "size_mb": size_fp32},
            "ptq_dynamic": {"accuracy": acc_dyn, "size_mb": size_dyn},
            "ptq_static": {"accuracy": acc_static, "size_mb": size_static}
        }

    # Save results
    with open("results/ptq_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to results/ptq_results.json")
```

### 4.2 Run PTQ

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/quantize_ptq.py 2>&1 | tee logs/phase4_ptq.log
```

### 4.3 Verify PTQ results

```bash
cat results/ptq_results.json
```

Save output to `logs/phase4_ptq_verification.txt`.

---

## PHASE 5: Quantization-Aware Training (QAT)

### 5.1 Create QAT script

Create file `scripts/quantize_qat.py`:

```python
#!/usr/bin/env python3
"""
Quantization-Aware Training (QAT) for PyTorch models.
Fine-tunes FP32 model with fake quantization nodes, then exports to INT8 ONNX.
"""

import os
import sys
import json
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import get_model, CONFIG


def qat_train(model_name, epochs=10, lr=1e-5):
    """Run QAT fine-tuning on pretrained model."""

    device = CONFIG["device"]
    model = get_model(model_name, num_classes=4, pretrained=False)

    # Load FP32 weights
    weights_path = f"models/fp32/{model_name}/best_model.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Prepare QAT
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    model_prepared = torch.quantization.prepare_qat(model, inplace=False)

    # Data loaders
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder("data/split/train", train_transform)
    val_dataset = datasets.ImageFolder("data/split/val", eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # QAT fine-tuning
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nQAT fine-tuning {model_name} for {epochs} epochs...")

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Train
        model_prepared.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            outputs = model_prepared(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model_prepared.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model_prepared(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best QAT model
            qat_model = torch.quantization.convert(model_prepared.eval(), inplace=False)
            torch.save(qat_model.state_dict(), f"models/qat/{model_name}_qat.pth")

    # Convert to quantized model
    qat_model = torch.quantization.convert(model_prepared.eval(), inplace=False)

    # Export to ONNX
    os.makedirs("models/qat", exist_ok=True)
    dummy_input = torch.randn(1, 3, 224, 224)
    output_path = f"models/qat/{model_name}_qat.onnx"

    # For QAT export, we need to use the quantized model's script
    # Alternative: export FP32 with quantization metadata
    torch.onnx.export(
        qat_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        input_names=["input"],
        output_names=["output"]
    )

    print(f"\nQAT complete for {model_name}")
    print(f"  Best val_acc: {best_val_acc:.4f}")
    print(f"  Model saved: {output_path}")

    return best_val_acc


if __name__ == "__main__":
    models = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]
    results = {}

    for model_name in models:
        print(f"\n{'#'*60}")
        print(f"# QAT: {model_name}")
        print(f"{'#'*60}")

        best_acc = qat_train(model_name, epochs=10, lr=1e-5)
        results[model_name] = {"qat_val_acc": best_acc}

    with open("results/qat_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nQAT results saved to results/qat_results.json")
```

### 5.2 Run QAT

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/quantize_qat.py 2>&1 | tee logs/phase5_qat.log
```

### 5.3 Verify QAT results

```bash
cat results/qat_results.json
```

Save output to `logs/phase5_qat_verification.txt`.

---

## PHASE 6: Hardware Benchmarking

### 6.1 Create benchmark script

Create file `scripts/benchmark.py`:

```python
#!/usr/bin/env python3
"""
Benchmark ONNX models for latency, throughput, and model size.
Run on target hardware (Raspberry Pi 4, CPU, etc.)
"""

import os
import json
import time
import numpy as np
import onnxruntime as ort


def benchmark_model(onnx_path, num_runs=1000, num_warmup=100):
    """Benchmark a single ONNX model."""

    session = ort.InferenceSession(onnx_path)
    input_meta = session.get_inputs()[0]
    input_shape = tuple(input_meta.shape)
    # Replace dynamic dimensions with 1
    input_shape = tuple(1 if isinstance(d, str) or d is None else d for d in input_shape)

    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {"input": dummy_input})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, {"input": dummy_input})
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    latencies = np.array(latencies)
    model_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)

    results = {
        "model_size_mb": round(model_size_mb, 2),
        "latency_mean_ms": round(float(latencies.mean()), 2),
        "latency_std_ms": round(float(latencies.std()), 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "throughput_fps": round(1000.0 / float(latencies.mean()), 2),
        "num_runs": num_runs,
        "input_shape": list(input_shape)
    }

    return results


def benchmark_all():
    """Benchmark all models (FP32, PTQ dynamic, PTQ static)."""

    model_configs = [
        ("mobilenetv2", "FP32", "models/onnx/mobilenetv2.onnx"),
        ("mobilenetv2", "PTQ_Dyn", "models/ptq/mobilenetv2_ptq_dynamic.onnx"),
        ("mobilenetv2", "PTQ_Static", "models/ptq/mobilenetv2_ptq_static.onnx"),
        ("shufflenetv2", "FP32", "models/onnx/shufflenetv2.onnx"),
        ("shufflenetv2", "PTQ_Dyn", "models/ptq/shufflenetv2_ptq_dynamic.onnx"),
        ("shufflenetv2", "PTQ_Static", "models/ptq/shufflenetv2_ptq_static.onnx"),
        ("efficientnet_b0", "FP32", "models/onnx/efficientnet_b0.onnx"),
        ("efficientnet_b0", "PTQ_Dyn", "models/ptq/efficientnet_b0_ptq_dynamic.onnx"),
        ("efficientnet_b0", "PTQ_Static", "models/ptq/efficientnet_b0_ptq_static.onnx"),
    ]

    all_results = {}

    for model_name, quant_type, path in model_configs:
        if not os.path.exists(path):
            print(f"SKIP: {path} not found")
            continue

        print(f"\nBenchmarking {model_name} ({quant_type})...")
        results = benchmark_model(path)

        key = f"{model_name}_{quant_type}"
        all_results[key] = results
        print(f"  Size: {results['model_size_mb']} MB")
        print(f"  Latency: {results['latency_mean_ms']} ms (±{results['latency_std_ms']})")
        print(f"  Throughput: {results['throughput_fps']} FPS")

    return all_results


if __name__ == "__main__":
    results = benchmark_all()

    # Save results
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Quant':<12} {'Size (MB)':<12} {'Latency (ms)':<15} {'FPS':<10}")
    print(f"{'='*80}")
    for key, res in results.items():
        parts = key.rsplit("_", 1)
        model = parts[0] if len(parts) == 1 else "_".join(parts[:-1])
        quant = parts[-1] if len(parts) > 1 else ""
        print(f"{model:<25} {quant:<12} {res['model_size_mb']:<12} {res['latency_mean_ms']:<15} {res['throughput_fps']:<10}")

    print(f"\nResults saved to results/benchmark_results.json")
```

### 6.2 Run benchmark

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/benchmark.py 2>&1 | tee logs/phase6_benchmark.log
```

### 6.3 Check system info

```bash
# Record hardware info for paper
echo "=== System Info ===" > logs/system_info.txt
echo "Date: $(date)" >> logs/system_info.txt
echo "Hostname: $(hostname)" >> logs/system_info.txt
uname -a >> logs/system_info.txt
echo "" >> logs/system_info.txt

# macOS specific
if command -v sysctl &> /dev/null; then
    echo "CPU: $(sysctl -n machdep.cpu.brand_string)" >> logs/system_info.txt
    echo "Cores: $(sysctl -n hw.ncpu)" >> logs/system_info.txt
    echo "Memory: $(($(sysctl -n hw.memsize) / 1024 / 1024 / 1024)) GB" >> logs/system_info.txt
fi

cat logs/system_info.txt
```

Save output to `logs/system_info.txt`.

---

## PHASE 7: Interpretability (Grad-CAM)

### 7.1 Create Grad-CAM script

Create file `scripts/gradcam.py`:

```python
#!/usr/bin/env python3
"""
Grad-CAM visualization for model interpretability.
Shows what features each architecture learns from fecal images.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


class GradCAM:
    """Simple Grad-CAM implementation."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        # Pool gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam / cam.max()

        # Resize to input size
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().numpy()

        return cam, output.softmax(dim=1).detach().numpy()


def get_target_layer(model_name, model):
    """Get the last convolutional layer for Grad-CAM."""
    if model_name == "mobilenetv2":
        return model.features[-1]
    elif model_name == "shufflenetv2":
        return model.conv5
    elif model_name == "efficientnet_b0":
        return model.features[-1]
    return None


def run_gradcam(model_name, model, class_names, data_dir="data/split/test", num_images=8):
    """Run Grad-CAM on sample images."""

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    dataset = datasets.ImageFolder(data_dir, transform)
    # Sample images from each class
    indices = []
    for cls_idx in range(len(class_names)):
        cls_indices = [i for i, (_, label) in enumerate(dataset) if label == cls_idx]
        indices.extend(cls_indices[:2])  # 2 images per class

    target_layer = get_target_layer(model_name, model)
    gradcam = GradCAM(model, target_layer)

    fig, axes = plt.subplots(len(indices), 3, figsize=(12, 4 * len(indices)))

    for idx, (img_idx, (img_tensor, label)) in enumerate([(i, dataset[i]) for i in indices]):
        input_tensor = img_tensor.unsqueeze(0)
        cam, probs = gradcam.generate(input_tensor)

        # Original image
        original = img_tensor.permute(1, 2, 0).numpy()
        original = (original * np.array([0.229, 0.224, 0.225]) +
                   np.array([0.485, 0.456, 0.406]))
        original = np.clip(original, 0, 1)

        # Grad-CAM overlay
        heatmap = plt.cm.jet(cam)[:, :, :3]
        overlay = 0.5 * original + 0.5 * heatmap

        # Plot
        axes[idx, 0].imshow(original)
        axes[idx, 0].set_title(f"Original\nTrue: {class_names[label]}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(cam, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM")
        axes[idx, 1].axis("off")

        axes[idx, 2].imshow(overlay)
        pred_class = np.argmax(probs)
        axes[idx, 2].set_title(f"Overlay\nPred: {class_names[pred_class]} ({probs[0][pred_class]:.2f})")
        axes[idx, 2].axis("off")

    plt.suptitle(f"Grad-CAM: {model_name}", fontsize=16)
    plt.tight_layout()

    output_path = f"figures/gradcam_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    from train import get_model

    class_names = ["cocci", "healthy", "ncd", "salmo"]
    models_to_analyze = ["mobilenetv2", "shufflenetv2", "efficientnet_b0"]

    os.makedirs("figures", exist_ok=True)

    for model_name in models_to_analyze:
        print(f"\nGrad-CAM for {model_name}...")

        model = get_model(model_name, num_classes=4, pretrained=False)
        weights_path = f"models/fp32/{model_name}/best_model.pth"
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        model.eval()

        run_gradcam(model_name, model, class_names)
```

### 7.2 Run Grad-CAM

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/gradcam.py 2>&1 | tee logs/phase7_gradcam.log
```

### 7.3 Verify visualizations

```bash
ls -la figures/
# Should contain: gradcam_mobilenetv2.png, gradcam_shufflenetv2.png, gradcam_efficientnet_b0.png
```

---

## PHASE 8: Dataset Size Sensitivity

### 8.1 Create sensitivity script

Create file `scripts/dataset_sensitivity.py`:

```python
#!/usr/bin/env python3
"""
Dataset size sensitivity experiment.
Train models on 25%, 50%, 75%, 100% of data to analyze generalization.
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent))
from train import get_model, train_one_epoch, validate, CONFIG


def create_subset(data_dir, fraction, output_dir, seed=42):
    """Create a subset of the training data."""
    random.seed(seed)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    n = int(len(full_dataset) * fraction)
    indices = random.sample(range(len(full_dataset)), n)
    subset = Subset(full_dataset, indices)

    return subset, len(full_dataset)


def run_sensitivity_experiment():
    """Train each model at different data fractions."""

    fractions = [0.25, 0.50, 0.75, 1.00]
    models_to_test = ["mobilenetv2", "efficientnet_b0"]  # Focus on 2 key models
    results = {}

    for model_name in models_to_test:
        results[model_name] = {}

        for fraction in fractions:
            print(f"\n{'='*60}")
            print(f"{model_name} @ {fraction*100}% data")
            print(f"{'='*60}")

            # Load data
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            eval_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])

            full_train = datasets.ImageFolder("data/split/train", transform)
            val_dataset = datasets.ImageFolder("data/split/val", eval_transform)
            test_dataset = datasets.ImageFolder("data/split/test", eval_transform)

            # Create subset
            n = int(len(full_train) * fraction)
            indices = random.sample(range(len(full_train)), n)
            train_subset = Subset(full_train, indices)

            train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

            # Train model
            model = get_model(model_name, num_classes=4)
            device = CONFIG["device"]
            model = model.to(device)
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            best_val_acc = 0
            for epoch in range(30):  # Reduced epochs for sensitivity
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                         optimizer, device)
                val_loss, val_acc, _ = validate(model, val_loader, criterion, device)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            # Test accuracy
            test_loss, test_acc, _ = validate(model, test_loader, criterion, device)

            results[model_name][f"{fraction*100:.0f}%"] = {
                "train_samples": n,
                "val_acc": best_val_acc,
                "test_acc": test_acc
            }

            print(f"  Train samples: {n}")
            print(f"  Best val_acc: {best_val_acc:.4f}")
            print(f"  Test_acc: {test_acc:.4f}")

    return results


if __name__ == "__main__":
    results = run_sensitivity_experiment()

    with open("results/sensitivity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSensitivity results saved to results/sensitivity_results.json")
```

### 8.2 Run sensitivity experiment

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/dataset_sensitivity.py 2>&1 | tee logs/phase8_sensitivity.log
```

---

## PHASE 9: Generate All Figures

### 9.1 Create visualization script

Create file `scripts/generate_figures.py`:

```python
#!/usr/bin/env python3
"""
Generate all figures for the paper.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def load_results():
    """Load all experiment results."""
    results = {}
    for name in ["ptq", "benchmark", "sensitivity"]:
        path = f"results/{name}_results.json"
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
    return results


def plot_accuracy_comparison(results, output_dir="figures"):
    """Plot accuracy comparison across architectures and quantization methods."""
    if "ptq" not in results:
        return

    models = list(results["ptq"].keys())
    methods = ["fp32", "ptq_dynamic", "ptq_static"]
    method_labels = ["FP32", "PTQ Dynamic", "PTQ Static"]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (method, label) in enumerate(zip(methods, method_labels)):
        accs = [results["ptq"][m][method]["accuracy"] for m in models]
        ax.bar(x + i * width, accs, width, label=label)

    ax.set_xlabel("Architecture")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison: FP32 vs PTQ")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.set_ylim(0.8, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved: fig_accuracy_comparison.png")


def plot_pareto_frontier(results, output_dir="figures"):
    """Plot accuracy-latency Pareto frontier."""
    if "benchmark" not in results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {"mobilenetv2": "o", "shufflenetv2": "s", "efficientnet_b0": "^"}
    colors = {"FP32": "blue", "PTQ_Dyn": "green", "PTQ_Static": "red"}

    for key, res in results["benchmark"].items():
        parts = key.rsplit("_", 1)
        model = parts[0]
        quant = parts[-1] if len(parts) > 1 else ""

        ax.scatter(res["latency_mean_ms"], res["model_size_mb"],
                  marker=markers.get(model, "o"),
                  c=colors.get(quant, "gray"),
                  s=100, label=f"{model} ({quant})")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Model Size (MB)")
    ax.set_title("Accuracy-Latency-Size Pareto Frontier")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_pareto_frontier.png"), dpi=150)
    plt.close()
    print(f"Saved: fig_pareto_frontier.png")


def plot_sensitivity(results, output_dir="figures"):
    """Plot dataset size sensitivity curves."""
    if "sensitivity" not in results:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, data in results["sensitivity"].items():
        fractions = sorted([int(k.replace("%", "")) for k in data.keys()])
        test_accs = [data[f"{f}%"]["test_acc"] for f in fractions]
        ax.plot(fractions, test_accs, marker="o", label=model_name, linewidth=2)

    ax.set_xlabel("Training Data Fraction (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Dataset Size Sensitivity Analysis")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks([25, 50, 75, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_sensitivity.png"), dpi=150)
    plt.close()
    print(f"Saved: fig_sensitivity.png")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    results = load_results()

    plot_accuracy_comparison(results)
    plot_pareto_frontier(results)
    plot_sensitivity(results)

    print("\nAll figures generated!")
    print(f"Files in figures/: {os.listdir('figures')}")
```

### 9.2 Generate figures

```bash
cd ~/research/paper_poultry_edge
source .venv/bin/activate
python3 scripts/generate_figures.py 2>&1 | tee logs/phase9_figures.log
```

---

## PHASE 10: Final Results Compilation

### 10.1 Compile all results

```bash
cd ~/research/paper_poultry_edge
python3 << 'EOF'
import json
import os

print("=" * 70)
print("FINAL RESULTS COMPILATION")
print("=" * 70)

# Load all results
results = {}
for name in ["ptq", "benchmark", "sensitivity", "qat"]:
    path = f"results/{name}_results.json"
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)
        print(f"\n✓ Loaded {name}_results.json")
    else:
        print(f"\n✗ Missing {name}_results.json")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Model':<25} {'FP32 Acc':<12} {'PTQ Acc':<12} {'Size (MB)':<12} {'Latency (ms)':<15}")
print("-" * 70)

if "ptq" in results:
    for model_name, data in results["ptq"].items():
        fp32_acc = data.get("fp32", {}).get("accuracy", "N/A")
        ptq_acc = data.get("ptq_static", {}).get("accuracy", "N/A")
        size = data.get("ptq_static", {}).get("size_mb", "N/A")

        latency = "N/A"
        if "benchmark" in results:
            for key, bench in results["benchmark"].items():
                if model_name in key and "Static" in key:
                    latency = bench.get("latency_mean_ms", "N/A")

        print(f"{model_name:<25} {fp32_acc:<12.4f} {ptq_acc:<12.4f} {size:<12.2f} {latency:<15}")

# Save compilation
with open("results/final_compilation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nFinal compilation saved to results/final_compilation.json")
EOF
```

Save output to `logs/phase10_final_results.txt`.

### 10.2 Git commit

```bash
cd ~/research/paper_poultry_edge
git add .
git commit -m "Complete ML2 experiments: training, PTQ, QAT, benchmarking, Grad-CAM, sensitivity analysis"
```

---

## EXPECTED OUTPUT FILES

After all phases complete, you should have:

```
~/research/paper_poultry_edge/
├── data/
│   ├── raw/           # Downloaded dataset
│   └── split/         # train/val/test splits
├── models/
│   ├── fp32/          # Trained FP32 models
│   ├── onnx/          # ONNX exported models
│   ├── ptq/           # PTQ quantized models
│   └── qat/           # QAT quantized models
├── scripts/
│   ├── train.py
│   ├── export_onnx.py
│   ├── quantize_ptq.py
│   ├── quantize_qat.py
│   ├── benchmark.py
│   ├── gradcam.py
│   ├── dataset_sensitivity.py
│   └── generate_figures.py
├── results/
│   ├── ptq_results.json
│   ├── benchmark_results.json
│   ├── qat_results.json
│   ├── sensitivity_results.json
│   └── final_compilation.json
├── figures/
│   ├── gradcam_*.png
│   ├── fig_accuracy_comparison.png
│   ├── fig_pareto_frontier.png
│   └── fig_sensitivity.png
├── logs/
│   ├── phase*.txt     # Verification logs
│   └── system_info.txt
└── outline.md         # Paper outline
```

---

## TROUBLESHOOTING

### Common Issues

1. **"No module named 'torch'"**
   - Solution: `source .venv/bin/activate` first

2. **Dataset download fails**
   - Alternative: Download manually from https://zenodo.org/records/4628934

3. **ONNX export error**
   - Check opset version: try `opset_version=11` instead of 13

4. **QAT produces NaN**
   - Reduce learning rate to `1e-6`
   - Reduce QAT epochs to 5

5. **Out of memory**
   - Reduce batch size to 16 or 8
   - Use `num_workers=2`

---

## REPORTING

After each phase, report:
1. **Completion status** (success/failure)
2. **Key metrics** (accuracy, latency, etc.)
3. **Any errors encountered** and how they were resolved
4. **Log file paths** for verification

Do NOT skip phases. Do NOT assume success. Execute and verify.

---

*Instructions created: August 15, 2026*
*Target: Pi agent execution*
