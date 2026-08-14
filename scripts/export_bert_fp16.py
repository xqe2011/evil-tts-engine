#!/usr/bin/env python3
"""Export BERT hidden_states[-3] to FP32 ONNX, then convert to FP16.

ZH: hfl/chinese-roberta-wwm-ext-large
JP: ku-nlp/deberta-v2-large-japanese-char-wwm
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

from to_fp16 import to_fp16

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


class HiddenSlice(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return out.hidden_states[-3]


def export_hs(model_dir: Path, dummy: str, out_fp32: Path) -> None:
    print(f"loading {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    base = AutoModelForMaskedLM.from_pretrained(str(model_dir))
    base.eval()
    wrapper = HiddenSlice(base).eval()
    ids = tok(dummy, return_tensors="pt")
    out_fp32.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (ids["input_ids"], ids["attention_mask"]),
        str(out_fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "hidden": {0: "batch", 1: "seq"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Wrote FP32 {out_fp32} ({out_fp32.stat().st_size / 1e6:.1f} MB)")
    del wrapper, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["zh", "jp"], required=True)
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--fp32", type=Path, help="keep FP32 ONNX (default: temp, then delete)")
    args = ap.parse_args()

    if args.lang == "zh":
        model_dir = args.model_dir or Path(
            "/Users/xqe2011/Documents/Code/tarot/tts/_work/zh-bert-dl/models/"
            "hfl--chinese-roberta-wwm-ext-large/snapshots/master"
        )
        dummy = "你好"
        stem = "chinese_roberta_wwm_ext_large_hs"
    else:
        model_dir = args.model_dir
        if model_dir is None:
            from huggingface_hub import snapshot_download

            model_dir = Path(snapshot_download("ku-nlp/deberta-v2-large-japanese-char-wwm"))
        dummy = "こんにちは"
        stem = "deberta_v2_large_japanese_char_wwm_hs"

    fp32 = args.fp32 or Path(f"/tmp/{stem}.onnx")
    fp16 = MODELS / f"{stem}.fp16.onnx"
    export_hs(model_dir, dummy, fp32)
    to_fp16(fp32, fp16, force_fp16_initializers=True)
    if args.fp32 is None and fp32.exists():
        fp32.unlink()
        data = Path(str(fp32) + ".data")
        if data.exists():
            data.unlink()
        print(f"removed temp {fp32}")


if __name__ == "__main__":
    main()
