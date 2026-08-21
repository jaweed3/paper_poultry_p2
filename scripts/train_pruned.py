#!/usr/bin/env python3
"""
train_pruned.py — Paper 2 finetune after structured pruning (SSOT: config.yaml)
Flow: load pretrained FP32 (paper1) -> prune (DepGraph L1) -> finetune N epochs -> export ONNX -> PTQ -> nodecount

Reproducible: reads config.yaml for all hparams. Logs to wandb (if available) + local json.

Usage (full, on server):
  .venv/bin/python scripts/train_pruned.py --model mobilenetv2 --sparsity 0.3 --epochs 15
  .venv/bin/python scripts/train_pruned.py --model mobilenetv2 --sparsity 0.3 --epochs 2 --dry-train  # 2 epoch smoke test

W&B: set WANDB_API_KEY or run `wandb login` for online. Otherwise runs in disabled/offline mode (still logs locally).
"""
import argparse
import json
import sys
from pathlib import Path
import time

import yaml
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"

def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)

# ---------- model ----------
def get_model(name, num_classes=4):
    if name == "mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "shufflenetv2":
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(name)
    return m

def find_pretrained(model_name):
    candidates = [
        Path(f"/mnt/c/Users/MASTER CORE TI/project/poultry_paper/fp32/{model_name}.pth"),
        Path(f"/mnt/c/Users/MASTER CORE TI/project/poultry_paper/models/fp32/{model_name}.pth"),
        ROOT / f"models/fp32/{model_name}.pth",
        ROOT / f"../project/poultry_paper/fp32/{model_name}.pth",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None

# ---------- dataset (copy from train_pipeline.py) ----------
def get_transforms(train=True, img_size=224):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

class PoultryDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", transform=None, class_names=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples = []
        class_names = class_names or ["cocci", "healthy", "ncd", "salmo"]
        for cls_idx, cls_name in enumerate(class_names):
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.glob("*.*"):
                if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                    self.samples.append((str(p), cls_idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, label

# ---------- pruning ----------
def prune_model(model, example_inputs, sparsity: float, num_classes=4):
    import torch_pruning as tp
    model.eval()
    imp = tp.importance.MagnitudeImportance(p=1)
    ignored = []
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.out_features == num_classes:
            ignored.append(m)
    # ShuffleNet fallback
    is_shufflenet = False
    try:
        import torchvision.models as _m
        if isinstance(model, _m.shufflenetv2.ShuffleNet_V2):
            is_shufflenet = True
    except Exception:
        pass
    if is_shufflenet:
        print("[WARN] ShuffleNetV2 skip prune (DepGraph 1.6.1 bug) -> no-op")
        return model, None
    pruner = tp.pruner.MagnitudePruner(
        model, example_inputs=example_inputs, importance=imp,
        iterative_steps=1, ch_sparsity=sparsity, ignored_layers=ignored,
    )
    pruner.step()
    return model, pruner

def export_onnx(model, name, sparsity, opset=17):
    outdir = ROOT / f"onnx_pruned/s{int(sparsity*100)}"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"{name}.onnx"
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    try:
        torch.onnx.export(model, dummy, str(outpath),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset, dynamo=False)
    except TypeError:
        torch.onnx.export(model, dummy, str(outpath),
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset)
    return outpath

def main():
    cfg = load_cfg()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=cfg["models"]["names"], default=cfg["pruning"]["dryrun_model"])
    ap.add_argument("--sparsity", type=float, default=cfg["pruning"]["dryrun_sparsity"])
    ap.add_argument("--epochs", type=int, default=cfg["finetune"]["epochs"])
    ap.add_argument("--lr", type=float, default=cfg["finetune"]["lr"])
    ap.add_argument("--batch-size", type=int, default=cfg["finetune"]["batch_size"])
    ap.add_argument("--dry-train", action="store_true", help="2 epoch smoke test with subset")
    ap.add_argument("--no-wandb", action="store_true", help="disable wandb")
    ap.add_argument("--wandb-project", type=str, default="paper-poultry-p2")
    args = ap.parse_args()

    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    import numpy as np, random
    np.random.seed(seed); random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] model={args.model} sparsity={args.sparsity} epochs={args.epochs} lr={args.lr} device={device}")

    # wandb
    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"{args.model}-s{int(args.sparsity*100)}-e{args.epochs}",
                config={"model": args.model, "sparsity": args.sparsity, "epochs": args.epochs, "lr": args.lr, "seed": seed, "batch_size": args.batch_size},
                mode="online" if __import__("os").environ.get("WANDB_API_KEY") else "offline",
            )
            print(f"[wandb] {wandb_run.mode} run {wandb_run.name}")
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            use_wandb = False

    # model
    num_classes = cfg["models"]["num_classes"]
    model = get_model(args.model, num_classes=num_classes)
    ckpt = find_pretrained(args.model)
    if ckpt:
        print(f"[ckpt] loading {ckpt}")
        sd = torch.load(ckpt, map_location="cpu")
        # handle checkpoint wrapped in dict
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        try:
            model.load_state_dict(sd, strict=True)
            print("[ckpt] loaded")
        except Exception as e:
            print(f"[ckpt] load warning: {e} -> trying strict=False")
            model.load_state_dict(sd, strict=False)
    else:
        print("[ckpt] no pretrained found -> random init (dry-train only)")

    before = sum(p.numel() for p in model.parameters())
    example = torch.randn(1, 3, cfg["dataset"]["img_size"], cfg["dataset"]["img_size"])
    model, pruner = prune_model(model, example, sparsity=args.sparsity, num_classes=num_classes)
    after = sum(p.numel() for p in model.parameters())
    print(f"[prune] {before:,} -> {after:,} ({100*after/max(before,1):.1f}% remain)")
    if use_wandb:
        try: import wandb; wandb.log({"params_before": before, "params_after": after, "params_remain_pct": 100*after/max(before,1)})
        except: pass

    # data
    data_root = ROOT / "data/images"
    if not data_root.exists():
        # fallback to Windows path
        data_root = Path("/mnt/c/Users/MASTER CORE TI/project/poultry_paper/data/images")
    print(f"[data] root={data_root}")

    train_tf = get_transforms(train=True, img_size=cfg["dataset"]["img_size"])
    val_tf = get_transforms(train=False, img_size=cfg["dataset"]["img_size"])
    class_names = cfg["dataset"]["classes"]
    train_ds = PoultryDataset(data_root, split="train", transform=train_tf, class_names=class_names)
    val_ds = PoultryDataset(data_root, split="val", transform=val_tf, class_names=class_names)
    print(f"[data] train {len(train_ds)} val {len(val_ds)}")

    if args.dry_train:
        # subset for smoke test
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(200, len(train_ds))))
        val_ds = Subset(val_ds, range(min(100, len(val_ds))))
        args.epochs = min(args.epochs, 2)
        print(f"[dry-train] subset train {len(train_ds)} val {len(val_ds)} epochs {args.epochs}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0
    best_path = None
    history = []

    import copy
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0; correct = 0; total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} train")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=loss.item())
        train_loss = running_loss / max(total,1)
        train_acc = 100*correct / max(total,1)

        # val
        model.eval()
        v_loss = 0; v_correct = 0; v_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                loss = criterion(out, labels)
                v_loss += loss.item() * imgs.size(0)
                _, pred = out.max(1)
                v_correct += pred.eq(labels).sum().item()
                v_total += labels.size(0)
        val_loss = v_loss / max(v_total,1)
        val_acc = 100*v_correct / max(v_total,1)
        scheduler.step()
        lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else args.lr

        print(f"Epoch {epoch+1}: train {train_loss:.4f}/{train_acc:.1f}% val {val_loss:.4f}/{val_acc:.1f}% lr {lr:.6f}")
        history.append({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": lr})
        if use_wandb:
            try: import wandb; wandb.log({"epoch": epoch+1, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": lr})
            except: pass
        if val_acc > best_acc:
            best_acc = val_acc
            best_wts = copy.deepcopy(model.state_dict())
            ckpt_dir = ROOT / f"models/pruned/s{int(args.sparsity*100)}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            best_path = ckpt_dir / f"{args.model}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> new best {best_acc:.2f}% saved {best_path}")

    model.load_state_dict(best_wts)
    # final save
    out_dir = ROOT / f"models/pruned/s{int(args.sparsity*100)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{args.model}_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"[done] best val {best_acc:.2f}% final {final_path}")

    # export + PTQ
    onnx_path = export_onnx(model.cpu(), args.model, args.sparsity, opset=cfg["quantization"]["export"]["opset"])
    print(f"[onnx] {onnx_path} {onnx_path.stat().st_size/1024/1024:.2f} MB")
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        int8_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
        quantize_dynamic(model_input=str(onnx_path), model_output=str(int8_path), weight_type=QuantType.QInt8)
        print(f"[ptq] {int8_path} {int8_path.stat().st_size/1024/1024:.2f} MB")
    except Exception as e:
        print(f"[ptq] failed: {e}")
        int8_path = None

    # save local json (always, wandb or not)
    meta = {
        "model": args.model, "sparsity": args.sparsity, "epochs": args.epochs, "lr": args.lr,
        "seed": seed, "params_before": before, "params_after": after,
        "best_val_acc": round(best_acc,2), "history": history,
        "ckpt": str(best_path) if best_path else None,
        "onnx": str(onnx_path), "int8": str(int8_path) if int8_path else None,
    }
    out_json = ROOT / f"results/finetune_{args.model}_s{int(args.sparsity*100)}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f: json.dump(meta, f, indent=2)
    print(f"[meta] {out_json}")
    if use_wandb:
        try: import wandb; wandb.finish()
        except: pass

if __name__ == "__main__":
    main()
