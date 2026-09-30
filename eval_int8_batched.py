#!/usr/bin/env python3
"""Batched FP32 vs INT8 ONNX accuracy - fast version."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import onnxruntime as ort
from collections import Counter

PROJECT = Path(__file__).resolve().parent  # repo root (absolute build path redacted for blind review)
DATA_TEST = PROJECT / "data" / "images" / "test"
MODELS = {
    "mobilenetv2": ("mobilenetv2_fp32.onnx", "mobilenetv2_int8.onnx"),
    "shufflenetv2": ("shufflenetv2_fp32.onnx", "shufflenetv2_int8.onnx"),
    "efficientnet_b0": ("efficientnet_b0_fp32.onnx", "efficientnet_b0_int8.onnx"),
}
CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
CLS2IDX = {c:i for i,c in enumerate(CLASS_NAMES)}
IMG_SIZE = 224
MEAN = np.array([0.485,0.456,0.406], dtype=np.float32)
STD = np.array([0.229,0.224,0.225], dtype=np.float32)
BATCH = 32

def preprocess_batch(paths):
    arrs = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        a = np.array(img, dtype=np.float32)/255.0
        a = (a - MEAN)/STD
        a = a.transpose(2,0,1)
        arrs.append(a)
    return np.stack(arrs).astype(np.float32)

def gather():
    samples=[]
    for cls in CLASS_NAMES:
        d=DATA_TEST/cls
        for f in d.iterdir():
            if f.suffix.lower() in (".jpg",".jpeg",".png",".bmp"):
                samples.append((str(f), CLS2IDX[cls]))
    print(f"Test samples: {len(samples)}")
    print(Counter(s[1] for s in samples))
    return samples

def eval_batched(onnx_path, samples):
    sess=ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp=sess.get_inputs()[0].name
    correct=0
    per_c=Counter(); per_t=Counter()
    # batch
    for i in range(0, len(samples), BATCH):
        chunk=samples[i:i+BATCH]
        paths=[c[0] for c in chunk]
        labels=[c[1] for c in chunk]
        x=preprocess_batch(paths)
        out=sess.run(None, {inp: x})[0]
        preds=np.argmax(out, axis=1)
        for pred,true in zip(preds, labels):
            if int(pred)==true:
                correct+=1
                per_c[true]+=1
            per_t[true]+=1
        if i % (BATCH*10)==0:
            print(f"  {i}/{len(samples)} acc so far {correct/max(1,i+BATCH)*100:.1f}%", flush=True)
    acc=correct/len(samples)*100
    per_acc={CLASS_NAMES[k]: round(per_c[k]/per_t[k]*100,2) if per_t[k] else 0 for k in range(len(CLASS_NAMES))}
    return acc, per_acc

def main():
    samples=gather()
    res={}
    for name,(fp32_file,int8_file) in MODELS.items():
        fp32=PROJECT/"models"/"onnx"/fp32_file
        int8=PROJECT/"models"/"ptq"/int8_file
        print(f"\n=== {name} ===")
        print(f" FP32 {fp32} exists={fp32.exists()}")
        print(f" INT8 {int8} exists={int8.exists()}")
        fp32_acc, fp32_per = eval_batched(fp32, samples)
        print(f" FP32 acc {fp32_acc:.4f} {fp32_per}")
        int8_acc, int8_per = eval_batched(int8, samples)
        print(f" INT8 acc {int8_acc:.4f} {int8_per}")
        res[name]={"fp32_acc": round(fp32_acc,4), "fp32_per_class": fp32_per, "int8_acc": round(int8_acc,4), "int8_per_class": int8_per, "drop": round(fp32_acc-int8_acc,4)}
    out=PROJECT/"results"/"int8_real_accuracy.json"
    with open(out,"w") as f: json.dump(res,f,indent=2)
    print(f"\nSaved {out}")
    print(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
