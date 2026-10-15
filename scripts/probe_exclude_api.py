"""Probe: how does ORT 1.29 quantize_static interpret nodes_to_exclude?

Checks whether it expects ONNX node names or initializer/output tensor names,
so scripts/ablate_layers.py passes the right thing.
"""
import inspect
import onnx
from onnxruntime.quantization import quantize_static
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer

print("=== quantize_static docstring (exclude part) ===")
doc = quantize_static.__doc__ or ""
for line in doc.split("\n"):
    if "exclude" in line.lower():
        print("  ", line.strip()[:170])

print("\n=== QDQQuantizer.__init__ signature ===")
try:
    sig = inspect.signature(QDQQuantizer.__init__)
    print("  ", sig)
except Exception as e:
    print("  ERR", e)

print("\n=== where nodes_to_exclude is consumed ===")
src = inspect.getsource(QDQQuantizer)
for i, line in enumerate(src.split("\n")):
    if "exclude" in line.lower():
        print(f"  {line.strip()[:170]}")

print("\n=== ONNX graph sample (efficientnet_b0 prepped) ===")
m = onnx.load("/home/master_core_ti/paper_poultry_p2/models/onnx/"
              "efficientnet_b0_gate0_prepped.onnx")
conv_nodes = [n for n in m.graph.node if n.op_type == "Conv"]
print("  total nodes:", len(m.graph.node))
print("  Conv nodes:", len(conv_nodes))
print("  first 5 Conv node names:")
for n in conv_nodes[:5]:
    grp = 1
    for a in n.attribute:
        if a.name == "group":
            grp = a.i
    print(f"    name={n.name!r} group={grp}")

depthwise = [n for n in conv_nodes
             if any(a.name == "group" and a.i > 1 for a in n.attribute)]
print(f"  Conv with group>1 (depthwise): {len(depthwise)}")
for n in depthwise[:5]:
    print(f"    name={n.name!r}")