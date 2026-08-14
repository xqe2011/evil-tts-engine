#!/usr/bin/env python3
"""Export ku-nlp/deberta-v2-large-japanese-char-wwm hidden_states[-3] to ONNX (+ optional int8).

Matches Bert-VITS2 V220 japanese_bert.py slice (1024-d) and evil-tts-engine EN/ZH exports.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForMaskedLM, AutoTokenizer


class HiddenSlice(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        return out.hidden_states[-3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model-dir",
        type=Path,
        help="HF model dir with config + weights (default: download ku-nlp/deberta-v2-large-japanese-char-wwm)",
    )
    ap.add_argument(
        "--out-fp32",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "deberta_v2_large_japanese_char_wwm_hs.onnx",
    )
    ap.add_argument(
        "--out-int8",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "models"
        / "deberta_v2_large_japanese_char_wwm_hs.int8.onnx",
    )
    ap.add_argument("--skip-int8", action="store_true")
    args = ap.parse_args()

    model_dir = args.model_dir
    if model_dir is None:
        from huggingface_hub import snapshot_download

        model_dir = Path(
            snapshot_download("ku-nlp/deberta-v2-large-japanese-char-wwm")
        )
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    base = AutoModelForMaskedLM.from_pretrained(str(model_dir))
    base.eval()
    wrapper = HiddenSlice(base)
    wrapper.eval()

    dummy = tok("こんにちは", return_tensors="pt")
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    args.out_fp32.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        str(args.out_fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "hidden": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Wrote FP32 {args.out_fp32} ({args.out_fp32.stat().st_size / 1e6:.1f} MB)")

    if not args.skip_int8:
        quantize_dynamic(str(args.out_fp32), str(args.out_int8), weight_type=QuantType.QInt8)
        print(f"Wrote INT8 {args.out_int8} ({args.out_int8.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
