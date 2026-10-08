"""Find the API that actually works for per-layer selective QDQ in ORT 1.29.

nodes_to_exclude is ignored for weight quantization. Candidates:
  1. nodes_to_quantize  -- explicit allowlist of Conv nodes to quantize
  2. extra_options NodeQuantConfig with tensor_quant_overrides
  3. quant_format QOperator (not QDQ)
"""
import hashlib
import os

import numpy as np
import onnx
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)

ROOT = "/home/master_core_ti/paper_poultry_p2"
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


def census(path, tag):
    m = onnx.load(path)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    qdq = sum(v for k, v in ops.items() if "Quantize" in k or "Dequantize" in k)
    convs = sum(1 for n in m.graph.node if n.op_type == "Conv")
    # count per-channel scales: scale tensors whose dim0 == weight out_channels
    init = {t.name: onnx.numpy_helper.to_array(t) for t in m.graph.initializer}
    scal = [t for t in init.values() if t.ndim == 1 and t.size > 8]
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()[:10]
    print(f"  {tag:<40} nodes={len(m.graph.node):<5} Conv={convs:<4} "
          f"Q/DQ={qdq:<5} vec_scales={len(scal):<4} md5={md5}")
    return md5, len(m.graph.node), qdq


m = onnx.load(PREP)
convs = [n.name for n in m.graph.node if n.op_type == "Conv"]
dw = [n.name for n in m.graph.node if n.op_type == "Conv"
      and any(a.name == "group" and a.i > 1 for a in n.attribute)]
nonconv = [n.name for n in m.graph.node if n.op_type != "Conv"]
all_names = [n.name for n in m.graph.node]
print(f"conv={len(convs)} dw={len(dw)} nonconv={len(nonconv)}")

print("\n=== 1. nodes_to_quantize = dw only ===")
try:
    quantize_static(
        model_input=PREP, model_output="/tmp/p1.onnx",
        calibration_data_reader=R(), quant_format=QuantFormat.QDQ,
        per_channel=False, activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=dw)
    census("/tmp/p1.onnx", "nodes_to_quantize=dw")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:180]}")

print("\n=== 2. nodes_to_quantize = convs, per_channel=True ===")
try:
    quantize_static(
        model_input=PREP, model_output="/tmp/p2.onnx",
        calibration_data_reader=R(), quant_format=QuantFormat.QDQ,
        per_channel=True, activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=convs)
    census("/tmp/p2.onnx", "nodes_to_quantize=convs pc")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:180]}")

print("\n=== 3. nodes_to_quantize = nonconv (expect ~0 QDQ) ===")
try:
    quantize_static(
        model_input=PREP, model_output="/tmp/p3.onnx",
        calibration_data_reader=R(), quant_format=QuantFormat.QDQ,
        per_channel=False, activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=nonconv)
    census("/tmp/p3.onnx", "nodes_to_quantize=nonconv")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:180]}")

print("\n=== 4. QOperator + nodes_to_quantize=dw ===")
try:
    quantize_static(
        model_input=PREP, model_output="/tmp/p4.onnx",
        calibration_data_reader=R(), quant_format=QuantFormat.QOperator,
        per_channel=False, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.MinMax,
        nodes_to_quantize=dw)
    census("/tmp/p4.onnx", "QOperator nodes_to_quantize=dw")
except Exception as e:
    print(f"  FAIL: {type(e).__name__}: {str(e)[:180]}")