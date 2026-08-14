#!/usr/bin/env python3
"""Convert FP32 ONNX graphs to mixed-precision FP16 (I/O stays original types)."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16

ELEM = {
    1: "f32",
    2: "u8",
    3: "i8",
    6: "i32",
    7: "i64",
    10: "f16",
    11: "f64",
}


def summarize(path: Path) -> str:
    m = onnx.load(str(path), load_external_data=False)
    types = Counter(init.data_type for init in m.graph.initializer)
    pretty = {ELEM.get(k, str(k)): v for k, v in types.items()}
    return f"{path.name}: {path.stat().st_size / 1e6:.1f}MB inits={pretty}"


def fix_range_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Range does not accept float16. Restore float16 Constant/Cast inputs to float32."""
    producers: dict[str, onnx.NodeProto] = {}
    for n in model.graph.node:
        for o in n.output:
            producers[o] = n

    for n in model.graph.node:
        if n.op_type != "Range":
            continue
        for inp in n.input:
            prod = producers.get(inp)
            if prod is None:
                continue
            if prod.op_type == "Constant":
                for attr in prod.attribute:
                    if attr.name != "value":
                        continue
                    t = attr.t
                    if t.data_type == TensorProto.FLOAT16:
                        arr = numpy_helper.to_array(t).astype(np.float32)
                        attr.t.CopyFrom(numpy_helper.from_array(arr, t.name))
            elif prod.op_type == "Cast":
                for attr in prod.attribute:
                    if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                        attr.i = TensorProto.FLOAT
    return model


def to_fp16(
    src: Path,
    dst: Path,
    *,
    force_fp16_initializers: bool = False,
    disable_shape_infer: bool = False,
    extra_op_block: list[str] | None = None,
) -> None:
    block = list(DEFAULT_OP_BLOCK_LIST)
    for op in extra_op_block or []:
        if op not in block:
            block.append(op)
    print(
        f"FP16 {src} → {dst} (force_init={force_fp16_initializers} "
        f"no_shape_infer={disable_shape_infer})"
    )
    model = convert_float_to_float16(
        str(src),
        keep_io_types=True,
        disable_shape_infer=disable_shape_infer,
        force_fp16_initializers=force_fp16_initializers,
        op_block_list=block,
    )
    model = fix_range_fp16(model)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    print("  wrote", summarize(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument(
        "--force-init",
        action="store_true",
        help="convert all float initializers to fp16 (breaks Range/etc. on some graphs)",
    )
    ap.add_argument(
        "--shape-infer",
        action="store_true",
        help="run ONNX shape/type inference before conversion",
    )
    args = ap.parse_args()
    to_fp16(
        args.src,
        args.dst,
        force_fp16_initializers=args.force_init,
        disable_shape_infer=not args.shape_infer,
    )


if __name__ == "__main__":
    main()
