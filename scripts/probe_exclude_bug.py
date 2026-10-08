"""Isolate the nodes_to_exclude bug: does excluding ALL node names kill quantization?"""
import json
import os
import onnx
import numpy as np
from onnxruntime.quantization import (quantize_static, quant_pre_process,
                                      CalibrationMethod, QuantFormat, QuantType,
                                      CalibrationDataReader)

ROOT = "/home/master_core_ti/paper_poultry_p2"
FP32 = f"{ROOT}/models/onnx/efficientnet_b0_gate0_fp32.onnx"
PREP = f"{ROOT}/models/onnx/efficientnet_b0_gate0_prepped.onnx"


class R(CalibrationDataReader):
    def __init__(self):
        self.n = 0
        self.done = False
        self.arr = np.random.rand(8, 3, 224, 224).astype(np.float32)

    def get_next(self):
        if self.done or self.n >= 100:
            self.done = True
            return None
        self.n += 8
        return {"input": self.arr}

    def rewind(self):
        self.n = 0
        self.done = False


def q(out, per_channel, exclude=None):
    extra = {"nodes_to_exclude": list(exclude)} if exclude is not None else {}
    quantize_static(
        model_input=PREP, model_output=out,
        calibration_data_reader=R(),
        quant_format=QuantFormat.QDQ, per_channel=per_channel,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options=extra,
    )
    m = onnx.load(out)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    qdq = sum(v for k, v in ops.items() if "Quantize" in k or "Dequantize" in k)
    convi = ops.get("ConvInteger", 0)
    print(f"  {os.path.basename(out):<42} nodes={len(m.graph.node):<5} "
          f"Q/DQ={qdq:<5} ConvInteger={convi:<4} "
          f"perchannel={'Y' if per_channel else 'N'}")
    return ops


import hashlib


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()[:12]


print("=== baseline: no exclusion ===")
a = q("/tmp/t_none_pc.onnx", True)
b = q("/tmp/t_none_pt.onnx", False)
print(f"  md5 pc={md5('/tmp/t_none_pc.onnx')} pt={md5('/tmp/t_none_pt.onnx')}")

m = onnx.load(PREP)
all_nodes = {n.name for n in m.graph.node}
convs = [n.name for n in m.graph.node if n.op_type == "Conv"]
dw = [n.name for n in m.graph.node if n.op_type == "Conv"
      and any(a2.name == "group" and a2.i > 1 for a2 in n.attribute)]
nonconv = all_nodes - set(convs)

print(f"\nnode census: total={len(all_nodes)} conv={len(convs)} dw={len(dw)} nonconv={len(nonconv)}")

print("\n=== exclude ALL non-Conv nodes (should still quantize convs per-channel) ===")
c = q("/tmp/t_nonconv_pc.onnx", True, exclude=nonconv)
print(f"  md5={md5('/tmp/t_nonconv_pc.onnx')}")

print("\n=== exclude ALL nodes (the bug case) ===")
d = q("/tmp/t_all_pc.onnx", True, exclude=all_nodes)
print(f"  md5={md5('/tmp/t_all_pc.onnx')}")

print("\n=== exclude only dw, per_channel on rest ===")
e = q("/tmp/t_excdw_pc.onnx", True, exclude=dw)
print(f"  md5={md5('/tmp/t_excdw_pc.onnx')}")

print("\n=== exclude everything except dw (should leave only dw quantized) ===")
f = q("/tmp/t_onlydw_pc.onnx", True, exclude=all_nodes - set(dw))
print(f"  md5={md5('/tmp/t_onlydw_pc.onnx')}")