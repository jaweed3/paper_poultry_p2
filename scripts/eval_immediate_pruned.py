import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["POULTRY_DATA_DIR"] = "data/images_sterile"

import train_pipeline as P
import torch_pruning as tp

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P.prepare_data()
    val_tf = P.get_transforms(train=False)
    test_ds = P.PoultryDataset(P.DATA_DIR, "test", val_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    teacher = P.get_model("mobilenetv2")
    teacher_ckpt = ROOT / "models/fp32/mobilenetv2_gate0.pth"
    teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    teacher = teacher.to(device)

    student = P.get_model("mobilenetv2")
    student.load_state_dict(torch.load(teacher_ckpt, map_location=device))
    student = student.to(device)

    example_input = torch.randn(1, 3, 224, 224).to(device)
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored_layers = [m for m in student.modules() if isinstance(m, nn.Linear)]
    pruner = tp.pruner.MagnitudePruner(
        student, example_inputs=example_input,
        importance=imp, pruning_ratio=0.30, ignored_layers=ignored_layers
    )
    pruner.step()

    student.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = student(x)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    preds, targets = np.array(preds), np.array(targets)
    acc = 100.0 * (preds == targets).sum() / len(targets)
    classes = ["cocci", "healthy", "ncd", "salmo"]
    print(f"IMMEDIATELY POST-PRUNING (s=0.30, 0 epoch fine-tuning): Top-1 = {acc:.2f}%")
    for i, c in enumerate(classes):
        m = targets == i
        rec = 100.0 * (preds[m] == i).sum() / m.sum()
        print(f"  {c} (n={m.sum()}): recall = {rec:.2f}% ({(preds[m]==i).sum()}/{m.sum()})")

if __name__ == "__main__":
    main()
