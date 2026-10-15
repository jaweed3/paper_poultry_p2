"""Why does nodes_to_exclude get ignored in ablate_layers.py?

Hypothesis: quant_pre_process fuses/reorders the graph so the node names we
collected from the ORIGINAL model no longer exist in the prepped model that is
actually quantized.

Test: compare Conv node names in original vs prepped, and check whether any of
our collected names survive.
"""
import onnx
from onnxruntime.quantization import quant_pre_process
from collections import Counter

ORIG = "/home/master_core_ti/paper_poultry_p2/models/onnx/efficientnet_b0_gate0_fp32.onnx"
PREP = "/home/master_core_ti/paper_poultry_p2/models/onnx/efficientnet_b0_gate0_prepped.onnx"

for tag, path in [("ORIGINAL", ORIG), ("PREPPED", PREP)]:
    m = onnx.load(path)
    convs = [n.name for n in m.graph.node if n.op_type == "Conv"]
    print(f"{tag}: total nodes={len(m.graph.node)} Conv={len(convs)}")
    for c in convs[:4]:
        print(f"    {c}")
    print()

mo = onnx.load(ORIG)
mp = onnx.load(PREP)
o = {n.name for n in mo.graph.node}
p = {n.name for n in mp.graph.node}
print(f"names in original: {len(o)}")
print(f"names in prepped : {len(p)}")
print(f"surviving by name: {len(o & p)}")
print(f"lost by fusion    : {len(o - p)}")
print(f"new in prepped    : {len(p - o)}")

# Do our collected group names survive?
import sys
sys.path.insert(0, "/home/master_core_ti/paper_poultry_p2/scripts")
from ablate_layers import classify_layers

groups = classify_layers(mo.graph)
for g, names in groups.items():
    if not names:
        continue
    surv = sum(1 for n in names if n in p)
    print(f"  {g:<11} {len(names):>3} collected, {surv:>3} survive into prepped")

print("\n=== does prepped still have separate SE convs? ===")
convs_p = [n.name for n in mp.graph.node if n.op_type == "Conv"]
fc = [c for c in convs_p if "/fc1/" in c or "/fc2/" in c]
print(f"  fc1/fc2 Conv nodes in prepped: {len(fc)}")
dw = [n for n in mp.graph.node if n.op_type == "Conv"
      and any(a.name == "group" and a.i > 1 for a in n.attribute)]
print(f"  depthwise Conv in prepped: {len(dw)}")