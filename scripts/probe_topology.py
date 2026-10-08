"""Inspect EfficientNet-B0 ONNX graph topology to design reliable DW/PW/SE grouping."""
import onnx
from collections import defaultdict

P = "/home/master_core_ti/paper_poultry_p2/models/onnx/efficientnet_b0_gate0_prepped.onnx"
m = onnx.load(P)
g = m.graph

print("op_type histogram:")
h = defaultdict(int)
for n in g.node:
    h[n.op_type] += 1
for k, v in sorted(h.items(), key=lambda x: -x[1]):
    print(f"  {k:<28} {v}")

# producer map: tensor -> node
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n

print("\n=== GlobalAveragePool / ReduceMean nodes and their consumers ===")
for n in g.node:
    if n.op_type in ("GlobalAveragePool", "ReduceMean"):
        print(f"  {n.op_type:<20} name={n.name!r}")
        for o in n.output:
            print(f"      out={o!r}")

print("\n=== Conv nodes whose 1x1 shape suggests SE (in==out channels) ===")
init = {t.name: t for t in g.initializer}


def shape_of(t):
    if t in init:
        return list(init[t].dims)
    return None


for n in g.node:
    if n.op_type != "Conv":
        continue
    conv_in = n.input[1] if len(n.input) > 1 else None
    dims = shape_of(conv_in)
    group = 1
    for a in n.attribute:
        if a.name == "group":
            group = a.i
    # SE convs are 1x1 with in==out (or in = C/4)
    if dims and len(dims) == 4 and dims[2] == 1 and dims[3] == 1:
        print(f"  1x1 Conv: {n.name!r}")
        print(f"      weight dims={dims} group={group}")

print("\n=== all Conv names + weight dims (first 25) ===")
cnt = 0
for n in g.node:
    if n.op_type != "Conv":
        continue
    conv_in = n.input[1] if len(n.input) > 1 else None
    dims = shape_of(conv_in)
    group = 1
    for a in n.attribute:
        if a.name == "group":
            group = a.i
    tag = "DW" if group > 1 else "  "
    print(f"  {tag} {n.name}")
    print(f"      w={dims} group={group}")
    cnt += 1
    if cnt >= 25:
        break