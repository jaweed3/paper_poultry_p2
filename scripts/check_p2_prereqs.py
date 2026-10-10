#!/usr/bin/env python3
"""check_p2_prereqs.py — cek bertahap, tanpa download, tanpa torch.

Phase 1: python + pip packages (onnxruntime, numpy, PIL, tqdm)
Phase 2: models (onnx fp32 + ptq int8)
Phase 3: data (data/images_sterile 5088/1337/1728)
Phase 4: Pi reachability (optional)

Run: python scripts/check_p2_prereqs.py
     python scripts/check_p2_prereqs.py --phase models
     POULTRY_DATA_DIR=data/images_sterile python scripts/check_p2_prereqs.py

Exit 0 jika phase yang diminta PASS, 1 jika FAIL.
"""
import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("POULTRY_DATA_DIR", str(ROOT / "data" / "images_sterile")))

CLASS_NAMES = ["cocci", "healthy", "ncd", "salmo"]
EXPECTED = {"train": 5088, "val": 1337, "test": 1728}

MODELS = {
    "mobilenetv2_fp32": "models/onnx/mobilenetv2_gate0_fp32.onnx",
    "mobilenetv2_int8": "models/ptq/mobilenetv2_gate0_int8.onnx",
    "shufflenetv2_fp32": "models/onnx/shufflenetv2_gate0_fp32.onnx",
    "shufflenetv2_int8": "models/ptq/shufflenetv2_gate0_int8.onnx",
    "efficientnet_b0_fp32": "models/onnx/efficientnet_b0_gate0_fp32.onnx",
    "efficientnet_b0_int8": "models/ptq/efficientnet_b0_gate0_int8.onnx",
}

def check_python():
    ok = sys.version_info >= (3, 10)
    return ok, f"python {sys.version.split()[0]}", "need >=3.10" if not ok else "OK"

def check_packages():
    results = {}
    for pkg, imp in [("onnxruntime", "onnxruntime"), ("numpy", "numpy"), ("PIL", "PIL"), ("tqdm", "tqdm")]:
        try:
            m = __import__(imp)
            ver = getattr(m, "__version__", "?")
            # pin check for onnxruntime
            if pkg == "onnxruntime" and ver != "1.29.0":
                results[pkg] = (False, ver, f"expected 1.29.0, got {ver} — pip install onnxruntime==1.29.0")
            else:
                results[pkg] = (True, ver, "OK")
        except ImportError as e:
            need = "WARN (optional)" if pkg == "tqdm" else "FAIL"
            results[pkg] = (pkg == "tqdm", "missing", f"{need}: {e} — pip install {pkg}")
    return results

def check_models():
    results = {}
    for name, rel in MODELS.items():
        p = ROOT / rel
        exists = p.exists()
        size = f"{p.stat().st_size/1024/1024:.1f} MB" if exists else "missing"
        hint = "OK" if exists else f"run ./scripts/fetch_p2_assets.sh models  OR  scp pi:~/p2bench/models/onnx/*.onnx models/onnx/"
        results[name] = (exists, size, hint)
    return results

def check_data():
    results = {}
    total = 0
    for split, exp in EXPECTED.items():
        n = 0
        for cls in CLASS_NAMES:
            d = DATA_DIR / split / cls
            if d.exists():
                n += len([f for f in d.glob("*.*") if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        ok = n == exp
        total += n
        hint = "OK" if ok else f"expected {exp}, got {n} — run ./scripts/fetch_p2_assets.sh data"
        results[split] = (ok, n, hint)
    results["_total"] = (total == 8153, total, "expected 8153" if total != 8153 else "OK")
    results["_data_dir"] = (DATA_DIR.exists(), str(DATA_DIR), "exists" if DATA_DIR.exists() else "missing")
    return results

def check_pi():
    import socket
    import subprocess
    # try tailscale + local IP
    hosts = [("pi-tailscale", "100.74.103.104"), ("pi-local", "192.168.119.43")]
    results = {}
    for label, ip in hosts:
        s = socket.socket()
        s.settimeout(1.5)
        try:
            s.connect((ip, 22))
            s.settimeout(1)
            try:
                banner = s.recv(64).decode(errors="ignore").strip()
            except:
                banner = "no banner"
            results[label] = (True, ip, banner[:40])
        except Exception as e:
            results[label] = (False, ip, str(e)[:60])
        finally:
            try:
                s.close()
            except:
                pass
    return results

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].startswith("--phase") else None
    if phase:
        # --phase=deps|models|data|pi
        phase = phase.split("=")[1] if "=" in phase else sys.argv[2] if len(sys.argv) > 2 else "all"

    print("=" * 60)
    print(f"CHECK P2 PREREQS  ROOT={ROOT}")
    print(f"DATA_DIR={DATA_DIR}")
    print("=" * 60)

    # Phase 1: python
    ok_py, ver, msg = check_python()
    print(f"\n[Phase 1] Python: {ver} -> {'PASS' if ok_py else 'FAIL'} ({msg})")

    # Phase 1b: packages
    pkgs = check_packages()
    print("[Phase 1] Packages:")
    for k, (ok, ver, hint) in pkgs.items():
        print(f"  {k:15s} {ver:12s} {'PASS' if ok else 'FAIL'} — {hint}")

    # Phase 2: models
    mods = check_models()
    print("\n[Phase 2] Models:")
    for k, (ok, sz, hint) in mods.items():
        print(f"  {k:28s} {sz:12s} {'PASS' if ok else 'FAIL'} — {hint}")

    # Phase 3: data
    data = check_data()
    print(f"\n[Phase 3] Data ({DATA_DIR}):")
    for k, (ok, n, hint) in data.items():
        print(f"  {k:15s} {str(n):12s} {'PASS' if ok else 'FAIL'} — {hint}")

    # Phase 4: pi (best-effort)
    pi = check_pi()
    print("\n[Phase 4] Pi reachability (best-effort):")
    for k, (ok, ip, banner) in pi.items():
        print(f"  {k:15s} {ip:15s} {'PASS' if ok else 'FAIL'} — {banner}")

    # summary
    # tqdm is optional
    pkg_ok = all(v[0] for k, v in pkgs.items() if k != "tqdm") and pkgs["onnxruntime"][0] and pkgs["numpy"][0] and pkgs["PIL"][0]
    model_ok = all(v[0] for v in mods.values())
    data_ok = data["_total"][0]
    all_ok = ok_py and pkg_ok and model_ok and data_ok

    print("\n" + "=" * 60)
    print(f"SUMMARY: python={'PASS' if ok_py else 'FAIL'}  packages={'PASS' if pkg_ok else 'FAIL'}  models={'PASS' if model_ok else 'FAIL'}  data={'PASS' if data_ok else 'FAIL'}")
    if not all_ok:
        print("NEXT: ./scripts/fetch_p2_assets.sh deps    # install light deps (2 sec)")
        print("      ./scripts/fetch_p2_assets.sh models   # download ONNX (30 MB)")
        print("      ./scripts/fetch_p2_assets.sh data     # scp from Pi or HF (8153 jpgs)")
    else:
        print("ALL PASS — ready to run: python scripts/mcnemar_mnv2_snv2.py")
    print("=" * 60)

    # exit code per phase if requested
    if phase == "deps":
        sys.exit(0 if (ok_py and pkg_ok) else 1)
    if phase == "models":
        sys.exit(0 if model_ok else 1)
    if phase == "data":
        sys.exit(0 if data_ok else 1)
    if phase == "pi":
        sys.exit(0 if any(v[0] for v in pi.values()) else 1)

if __name__ == "__main__":
    main()
